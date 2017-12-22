from docopt import docopt
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import lobpcg
import scipy.sparse
import time


def main():
    args = docopt("""
    Usage:
        learn_spectral_embeddings.py [options] <adjacency_matrix_path> <type_of_laplacian> <output_path>

    Options:
        --pow NUM          Every non-zero value in adjacency matrix will be scaled^{pow} [default: 1.0]
        --max_iter NUM     Maximum number of iterations of LOBPCG algorithm [default: 100]
        --dim NUM          Number of eigen-pairs to return [default: 500]
        --verbosity NUM    Verbosity level of LOBPCG solver [default: 0]
    """)

    start = time.time()
    print("Loading adjacency matrix, %f" % time.time())
    adjacency_matrix = load_npz(args["<adjacency_matrix_path>"])
    power = float(args["--pow"])
    if pow == 0.0:
        adjacency_matrix.data = np.ones_like(adjacency_matrix.data, dtype=np.float64)
    elif power < 1.0:
        adjacency_matrix.data = adjacency_matrix.data**power
    elif power > 1.0:
        raise NotImplementedError("We accept only power in [0,1] and it is %f" % power)

    type_of_laplacian = args["<type_of_laplacian>"]
    print("Building %s laplacian, %f" % (type_of_laplacian, time.time()))
    n = adjacency_matrix.shape[0]
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    L = D - adjacency_matrix

    rng = np.random.RandomState(0)
    dim = int(args["--dim"]) + 1
    init = rng.rand(n, dim)
    init[:, 0] = np.ones(n)
    B = None

    if type_of_laplacian == "random_walk_normalized":
        D_inv = scipy.sparse.spdiags(1.0/degrees, [0], n, n, format='csr')
        B = D_inv
    elif type_of_laplacian == "symmetric_normalized":
        degrees_sqrt = np.sqrt(degrees)
        D_inv_sqrt = scipy.sparse.spdiags(1.0 / degrees_sqrt, [0], n, n, format='csr')
        L = D_inv_sqrt.dot(L.dot(D_inv_sqrt))
        init[:, 0] = degrees_sqrt
    elif not type_of_laplacian == "unnormalized":
        raise NotImplementedError("The type %s of laplacian is not implemented" % type_of_laplacian)

    print("Solving for eigenvectors and eigenvalues, %f" % time.time())
    max_iter = int(args["--max_iter"])
    verbosity = int(args["--verbosity"])
    vals, vecs = lobpcg(L, X=init, B=B, maxiter=max_iter, largest=False, verbosityLevel=verbosity)

    postfix = "_%s_pow=%.2f_dim=%d" % (type_of_laplacian, power, dim)
    output_path = args["<output_path>"] + postfix

    np.save(output_path + ".vecs", vecs[:, 1:])
    np.save(output_path + ".vals", vals[1:])
    np.save(output_path + ".degrees", degrees)
    print("Time elapsed %f" % (time.time() - start))

if __name__ == '__main__':
    main()