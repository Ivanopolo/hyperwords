import numpy as np
import scipy.sparse
from scipy.sparse.linalg import minres, LinearOperator, eigsh


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


def estimate_rhoB(adjacency_matrix):
    print("Tuning rhoB estimation")
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    guessForFirstEigen = (degrees ** 2).mean() / degrees.mean() - 1
    errtol = 1e-2
    maxIter = 10

    err = 1
    iteration = 0
    rhoB = guessForFirstEigen
    print("Initial guess of rhoB is %f" % rhoB)
    while err > errtol and iteration < maxIter:
        iteration += 1
        print("Building matrices")
        BH = build_weighted_bethe_hessian(adjacency_matrix, rhoB)
        BHprime = build_weighted_bethe_hessian_derivative(adjacency_matrix, rhoB)

        sigma = 0
        op_inverse = lambda v: minres(BH, v, tol=1e-5)[0]
        OPinv = LinearOperator(matvec=op_inverse, shape=adjacency_matrix.shape, dtype=np.float64)

        print("Solving the eigenproblem")
        mu, x = eigsh(A=BH, M=BHprime, k=1, which='LM', sigma=sigma, OPinv=OPinv)
        mu = mu[0]
        print("mu is %f" % mu)
        err = abs(mu)
        rhoB -= mu
        print("Iteration %d, updated value of rhoB %f" % (iteration, rhoB))

    return rhoB
