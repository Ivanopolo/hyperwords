import time

import numpy as np
import scipy.sparse
from docopt import docopt
from scipy.sparse import load_npz
from scipy.sparse.linalg import lobpcg, eigsh

from ..utils.tools import build_weighted_bethe_hessian, estimate_rhoB, eigsh_slepc
from ..representations.matrix_serializer import load_vocabulary


def main():
    args = docopt("""
    Usage:
        learn_spectral_embeddings.py [options] <adjacency_matrix_path> <type_of_laplacian> <output_path>

    Options:
        --pow NUM          Every non-zero value in adjacency matrix will be scaled^{pow} [default: 1.0]
        --tol NUM          Digits of relative precision for eigenvalue decomposition [default: 8]
        --max_iter NUM     Maximum number of iterations of LOBPCG algorithm [default: 100]
        --dim NUM          Number of eigen-pairs to return [default: 500]
        --verbosity NUM    Verbosity level of LOBPCG solver [default: 0]
        --pmi              Turn adjacency matrix into PMI-based adjacency matrix
        --neg NUM          Negative sampling for PMI-based adjacency matrix [default: 1]
        --scale_weights    Scale weights of the adjacency matrix between 0 and 1
        --tune_rhoB        Solve quadratic eigenproblem to find better rhoB estimation
    """)

    start = time.time()
    print("Loading adjacency matrix, %f" % time.time())
    adjacency_matrix_path = args["<adjacency_matrix_path>"]
    adjacency_matrix = load_npz(adjacency_matrix_path + ".adjacency.npz")
    _, iw = load_vocabulary(adjacency_matrix_path + ".words.vocab")
    power = float(args["--pow"])
    if power <= 1.0:
        adjacency_matrix.data = adjacency_matrix.data**power
    elif power > 1.0 or power < 0.0:
        raise NotImplementedError("We accept only power in [0,1] and it is %f" % power)

    if args["--scale_weights"]:
        print("Scaling weights of the adjacency matrix")
        adjacency_matrix.data /= np.max(adjacency_matrix.data)

    n = adjacency_matrix.shape[0]
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    if args["--pmi"]:
        neg = int(args["--neg"])
        print("Building PMI matrix with negative sampling=%d" % neg)
        print("Number of non-zero elements is: %d" % adjacency_matrix.count_nonzero())
        total_count = degrees.sum()
        D_inv = scipy.sparse.spdiags(1.0 / degrees, [0], n, n, format='csr')
        adjacency_matrix = D_inv.dot(adjacency_matrix.dot(D_inv))
        adjacency_matrix.data = np.maximum(np.log(adjacency_matrix.data * total_count) - np.log(neg), 0)
        adjacency_matrix.eliminate_zeros()
        degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten() #Update degrees
        print("Number of non-zero elements is: %d" % adjacency_matrix.count_nonzero())

    type_of_laplacian = args["<type_of_laplacian>"]
    print("Building %s laplacian, %f" % (type_of_laplacian, time.time()))

    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    L = D - adjacency_matrix

    rng = np.random.RandomState(0)
    dim = int(args["--dim"]) + 1
    init = rng.rand(n, dim)
    init[:, 0] = np.ones(n)
    B = None
    preconditioner = None

    if type_of_laplacian == "unnormalized":
        preconditioner = scipy.sparse.spdiags(1.0/degrees, [0], n, n, format='csr')
    elif type_of_laplacian == "random_walk_normalized":
        D_inv = scipy.sparse.spdiags(1.0/degrees, [0], n, n, format='csr')
        B = D_inv
    elif type_of_laplacian == "symmetric_normalized":
        degrees_sqrt = np.sqrt(degrees)
        D_inv_sqrt = scipy.sparse.spdiags(1.0 / degrees_sqrt, [0], n, n, format='csr')
        L = D_inv_sqrt.dot(L.dot(D_inv_sqrt))
        init[:, 0] = degrees_sqrt
    elif type_of_laplacian == "bethe_hessian":
        r = np.sqrt((degrees ** 2).mean() / degrees.mean() - 1)

        if args["--tune_rhoB"]:
            r = np.sqrt(estimate_rhoB(adjacency_matrix))

        L = build_weighted_bethe_hessian(adjacency_matrix, r)
    else:
        raise NotImplementedError("The type %s of laplacian is not implemented" % type_of_laplacian)

    print("Solving for eigenvectors and eigenvalues, %f" % time.time())
    max_iter = int(args["--max_iter"])
    verbosity = int(args["--verbosity"])

    if type_of_laplacian == "bethe_hessian":
        ### Lanzcos algorithm for Bethe Hessian
        digits_of_precision = int(args["--tol"])
        tol = (1.0 / (10.0**digits_of_precision)) * n
        print("Requested tolerance is %f" % tol)
        #vals, vecs = eigsh(L, dim - 1, which='SA', tol=tol)
        eigsh_slepc(L, k=dim-1, tol=tol, max_iter=max_iter)

        ### LOBPCG learning
        # vals, vecs = lobpcg(L, M=preconditioner, X=init, B=B, maxiter=max_iter, largest=False, verbosityLevel=verbosity)
    else:
        ### LOBPCG learning
        vals, vecs = lobpcg(L, M=preconditioner, X=init, B=B, maxiter=max_iter, largest=False, verbosityLevel=verbosity)
        vals = vals[1:]
        vecs = vecs[:, 1:]

    postfix = "_%s_pow=%.2f_dim=%d" % (type_of_laplacian, power, dim)
    output_path = args["<output_path>"] + postfix

    np.save(output_path + ".vecs", vecs)
    np.save(output_path + ".vals", vals)
    np.save(output_path + ".degrees", degrees)

    with open(output_path + ".words.vocab", "w") as f:
        for i, w in enumerate(iw):
            f.write("%s\n" % w)

    print("Time elapsed %f" % (time.time() - start))


if __name__ == '__main__':
    main()
