import time

import numpy as np
import scipy.sparse
from docopt import docopt
from scipy.sparse import load_npz
from scipy.sparse.linalg import lobpcg, eigsh
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils.tools import build_weighted_bethe_hessian, estimate_rhoB
from ..representations.matrix_serializer import load_vocabulary


def main():
    args = docopt("""
    Usage:
        learn_spectral_embeddings.py [options] <adjacency_matrix_path> <type_of_laplacian> <output_path>

    Options:
        --pow NUM          Every non-zero value in adjacency matrix will be scaled^{pow} [default: 1.0]
        --max_iter NUM     Maximum number of iterations of LOBPCG algorithm [default: 100]
        --dim NUM          Number of eigen-pairs to return [default: 500]
        --verbosity NUM    Verbosity level of LOBPCG solver [default: 0]
        --pmi              Turn adjacency matrix into PMI-based adjacency matrix
        --neg NUM          Negative sampling for PMI-based adjacency matrix [default: 1]
        --scale_weights    Scale weights of the adjacency matrix between 0 and 1
        --tune_rhoB        Solve quadratic eigenproblem to find better rhoB estimation
    """)

    start = time.time()

    adjacency_matrix_path = args["<adjacency_matrix_path>"]
    adjacency_matrix = load_npz(adjacency_matrix_path + ".adjacency.npz")
    _, iw = load_vocabulary(adjacency_matrix_path + ".words.vocab")

    ### Build laplacian
    n = adjacency_matrix.shape[0]
    degrees = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    L = D - adjacency_matrix
    degrees_sqrt = np.sqrt(degrees)
    DH = scipy.sparse.spdiags(1.0 / degrees_sqrt, [0], n, n, format='csr')
    L = DH.dot(L.dot(DH))

    ### Build PETSc sparse matrix
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    inds = L.nonzero()

    for row_ind, col_ind in np.stack(inds, axis=1):
        value = L[row_ind, col_ind]
        assert value != 0
        A[row_ind, col_ind] = value

    A.assemble()

    ### Solve eigenvalue problem
    E = SLEPc.EPS()
    E.create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)




    print("Time elapsed %f" % (time.time() - start))


if __name__ == '__main__':
    main()
