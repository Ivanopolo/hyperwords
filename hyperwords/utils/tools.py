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


def build_weighted_bethe_hessian_derivative(adjacency_matrix, r):
    n = adjacency_matrix.shape[0]

    w = adjacency_matrix.data.copy()
    dt = - w * (r ** 2 + w ** 2)
    dt /= (r ** 2 - w ** 2) ** 2

    adjacency_matrix.data = - (2 * r * w ** 2) / (r ** 2 - w ** 2) ** 2
    bethe_der_diagonal = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(bethe_der_diagonal, [0], n, n, format='csr')

    adjacency_matrix.data = dt
    Hr_prime = D - adjacency_matrix
    return Hr_prime