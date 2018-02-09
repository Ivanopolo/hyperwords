import numpy as np
import scipy.sparse
import time
from scipy.sparse.linalg import minres, LinearOperator, eigsh
from petsc4py import PETSc
from slepc4py import SLEPc


def build_weighted_bethe_hessian(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]

    dt = A.data * r
    dt /= r ** 2 - A.data ** 2

    A.data = A.data ** 2 / (r ** 2 - A.data ** 2)
    bethe_diagonal = 1 + np.asarray(A.sum(axis=1), dtype=np.float64).flatten()
    D = scipy.sparse.spdiags(bethe_diagonal, [0], n, n, format='csr')

    A.data = dt
    return D, A


def build_weighted_bethe_hessian_derivative(adjacency_matrix, r):
    A = adjacency_matrix.copy()
    n = A.shape[0]

    w = A.data
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
        err = abs(mu) / rhoB
        rhoB -= mu
        print("Iteration %d, updated value of rhoB %f, relative error %f" % (iteration, rhoB, err))

    return rhoB


class MatrixOperator(object):

    def __init__(self, A):
        self.A = A.astype(PETSc.ScalarType)
        self.n_calls = 0

    def mult(self, A, x, y):
        xx = x.getArray(readonly=1)
        yy = y.getArray(readonly=0)
        yy[:] = self.A.dot(xx)
        self.n_calls += 1

def eigsh_slepc(A, k, tol, max_iter):

    ### Setup matrix operator
    n = A.shape[0]
    mat = MatrixOperator(A)
    A_operator = PETSc.Mat().createPython([n, n], mat)
    A_operator.setUp()

    ### Solve eigenproblem
    E = SLEPc.EPS()
    E.create()
    E.setOperators(A_operator)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setDimensions(k)
    E.setTolerances(tol, max_iter)
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    def monitor_fun(eps, iters, nconv, eigs, errors):
        print("Current iteration: %d, number of converged eigenvalues: %d" % (iters, nconv))

    E.setMonitor(monitor_fun)
    E.solve()
    print("Number of calls to Ax: %d" % mat.n_calls)

    ### Collect results
    print("")
    its = E.getIterationNumber()
    print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    print("NEV %d NCV %d MPD %d" % (nev, ncv, mpd))
    tol, maxit = E.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    print("Number of converged eigenpairs: %d" % nconv)
    nconv = min(nconv, k)

    if nconv < k:
        raise ZeroDivisionError("Failed to converge for requested number of k with maxiter=%d" % max_iter)

    vecs = np.zeros([n, nconv])
    vals = np.zeros(nconv)

    xr, tmp = A_operator.getVecs()
    xi, tmp = A_operator.getVecs()

    if nconv > 0:
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            vals[i] = k.real
            vecs[:, i] = xr

    return vals, vecs
