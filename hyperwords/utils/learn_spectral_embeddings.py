import time

import numpy as np
import scipy.sparse
from docopt import docopt
from scipy.sparse.linalg import lobpcg, eigsh

from ..utils.counts2svd import build_ppmi_matrix, load_adjacency_matrix
from ..utils.tools import build_weighted_bethe_hessian, estimate_rhoB, eigsh_slepc
from ..representations.matrix_serializer import load_vocabulary


def main():
    args = docopt("""
    Usage:
        learn_spectral_embeddings.py [options] <counts_path> <type_of_laplacian>

    Options:
        --pow NUM          Every non-zero value in adjacency matrix will be scaled^{pow} [default: 1.0]
        --tol NUM          Digits of relative precision for eigenvalue decomposition [default: 0.01]
        --max_iter NUM     Maximum number of iterations of LOBPCG algorithm [default: 100]
        --dim NUM          Number of eigen-pairs to return [default: 500]
        --verbosity NUM    Verbosity level of LOBPCG solver [default: 0]
        --pmi              Zero elements of adjacency matrix that are zero in positive PMI matrix
        --neg NUM          Negative sampling for PMI-based adjacency matrix [default: 1]
        --scale_weights    Scale weights of the adjacency matrix between 0 and 1
        --tune_rhoB        Solve quadratic eigenproblem to find better rhoB estimation
    """)

    start = time.time()
    print("Loading adjacency matrix, %f" % time.time())
    counts_path = args["<counts_path>"]
    adjacency_matrix = load_adjacency_matrix(counts_path)
    _, iw = load_vocabulary(counts_path + ".words.vocab")

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
        pmi_matrix = build_ppmi_matrix(adjacency_matrix, cds=0.75, neg=neg)
        zeros_mask = pmi_matrix.data == 0
        adjacency_matrix.data[zeros_mask] = 0
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

        D, A = build_weighted_bethe_hessian(adjacency_matrix, r)
        L = D - A
    else:
        raise NotImplementedError("The type %s of laplacian is not implemented" % type_of_laplacian)

    print("Solving for eigenvectors and eigenvalues, %f" % time.time())
    max_iter = int(args["--max_iter"])
    verbosity = int(args["--verbosity"])

    tol = float(args["--tol"])
    print("Requested tolerance is %f" % tol)

    if type_of_laplacian == "bethe_hessian":
        ### Lanzcos algorithm for Bethe Hessian
        #vals, vecs = eigsh(L, dim - 1, which='SA', tol=tol)
        vals, vecs = eigsh_slepc(L, k=dim-1, tol=tol, max_iter=max_iter)

        ### LOBPCG learning
        # vals, vecs = lobpcg(L, M=preconditioner, X=init[:, 1:], B=B, maxiter=max_iter, largest=False, verbosityLevel=verbosity)
    else:
        ### LOBPCG learning
        vals, vecs = lobpcg(L, M=preconditioner, X=init, B=B, maxiter=max_iter, largest=False, verbosityLevel=verbosity)
        vals = vals[1:]
        vecs = vecs[:, 1:]

    postfix = "_%s_pow=%.2f_dim=%d_tol=%f" % (type_of_laplacian, power, dim, tol)
    output_path = counts_path + postfix

    np.save(output_path + ".vecs", vecs)
    np.save(output_path + ".vals", vals)
    np.save(output_path + ".degrees", degrees)

    with open(output_path + ".words.vocab", "w") as f:
        for i, w in enumerate(iw):
            f.write("%s\n" % w)

    print("Time elapsed %f" % (time.time() - start))


if __name__ == '__main__':
    main()
