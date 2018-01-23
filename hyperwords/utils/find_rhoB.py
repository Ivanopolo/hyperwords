import time

import numpy as np
import scipy.sparse
from docopt import docopt
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh, minres, LinearOperator


def main():
    args = docopt("""
    Usage:
        find_rhoB.py [options] <adjacency_matrix_path>
    """)

    start = time.time()
    print("Loading adjacency matrix, %f" % time.time())
    adjacency_matrix_path = args["<adjacency_matrix_path>"]
    adjacency_matrix = load_npz(adjacency_matrix_path + ".adjacency.npz")
    adjacency_matrix.data = adjacency_matrix.data ** 0.0

    n = adjacency_matrix.shape[0]
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    I = scipy.sparse.eye(n, format='csr')

    def buildBH(r):
        return (r ** 2 - 1) * I - r * adjacency_matrix + D

    def buildBHprime(r):
        return 2 * r * I - adjacency_matrix

    guessForFirstEigen = (degrees ** 2).mean() / degrees.mean() - 1
    errtol = 1e-10
    maxIter = 10

    err = 1
    iteration = 0
    rhoB = guessForFirstEigen
    print("Initial guess is %f" % rhoB)
    while err > errtol and iteration < maxIter:
        iteration += 1
        print("Building matrices")
        BH = buildBH(rhoB)
        BHprime = buildBHprime(rhoB)

        sigma = 0
        op_inverse = lambda v: minres(BH, v, tol=1e-5)
        OPinv = LinearOperator(matvec=op_inverse[0], shape=adjacency_matrix.shape, dtype=np.float64)

        print("Solving the eigenproblem")
        mu, x = eigsh(A=BH, M=BHprime, k=1, which='LM', sigma=sigma, OPinv=OPinv)
        print("mu is %f" % mu)
        err = abs(mu)
        rhoB -= mu
        print("updated value of rhoB %f" % rhoB)
        print(err, iteration)

    print("Time elapsed %f" % (time.time() - start))


if __name__ == '__main__':
    main()