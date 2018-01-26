import numpy as np
import scipy.sparse


### Correct
def build_weighted_bethe_hessian(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]

    dt = A.data.copy() * r
    dt /= r ** 2 - A.data.copy() ** 2

    A.data = A.data ** 2 / (r ** 2 - A.data ** 2)
    bethe_diagonal = 1 + np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(bethe_diagonal, [0], n, n, format='csr')

    A.data = dt
    Hr = D - A
    return Hr


### Correct
def build_weighted_bethe_hessian_direct(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]
    I = scipy.sparse.eye(n, format='csr')
    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    Hr = I + D / (r**2-1) - r/(r**2-1) * A
    return Hr


def build_weighted_bethe_hessian_derivative(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]

    w = A.data.copy()
    dt = - w * (r ** 2 + w ** 2)
    dt /= (r ** 2 - w ** 2) ** 2

    A.data = - (2 * r * w ** 2) / (r ** 2 - w ** 2) ** 2
    bethe_der_diagonal = np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(bethe_der_diagonal, [0], n, n, format='csr')

    A.data = dt
    Hr_prime = D - A
    return Hr_prime

### Correct
def build_weighted_bethe_hessian_derivative_direct(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]
    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    Hr_prime = - 2*r / (r**2 - 1)**2 * D + (r**2 + 1) / (r**2 - 1)**2 * A
    return Hr_prime
