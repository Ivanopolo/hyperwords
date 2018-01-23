import numpy as np
import scipy.sparse


def build_weighted_bethe_hessian(adjacency_matrix, r):
    n = adjacency_matrix.shape[0]

    dt = adjacency_matrix.data * r
    dt /= r ** 2 - adjacency_matrix.data ** 2

    adjacency_matrix.data = adjacency_matrix.data ** 2 / (r ** 2 - adjacency_matrix.data ** 2)
    bethe_diagonal = 1 + np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(bethe_diagonal, [0], n, n, format='csr')

    adjacency_matrix.data = dt
    Hr = D - adjacency_matrix
    return Hr
