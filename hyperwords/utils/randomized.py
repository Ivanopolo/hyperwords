import logging

import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse import issparse
from sklearn.utils import check_random_state


def safe_sparse_dot(a, b, dense_output=False):
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output:
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def randomized_range_finder(A, size, n_iter,
                            power_iteration_normalizer,
                            random_state):

    # Generating normal random vectors with shape: (A.shape[0], size)
    logging.info("Generating random matrix")
    Q = random_state.normal(size=(A.shape[0], size))

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q

    # TODO: disacard factors with close to zero length
    for i in range(n_iter):
        logging.info("%d iteration of power method, power normalizer used: %s" % (i+1, power_iteration_normalizer))
        if power_iteration_normalizer == 'none':
            Q = safe_sparse_dot(A, Q)
            Q = safe_sparse_dot(A.T, Q)
        elif power_iteration_normalizer == 'LU':
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
            Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
            Q, _ = linalg.qr(safe_sparse_dot(A.T, Q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis

    # TODO: try double orthogonalization
    logging.info("Producing orthogonal basis")
    Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
    return Q


def randomized_eigh(M, n_components, n_oversamples=10, n_iter=0, power_iteration_normalizer='QR', random_state=0):
    logging.info("Starting randomized SVD with %d components, %d oversamples, %d power iterations" %
                 (n_components, n_oversamples, n_iter))
    assert M.shape[0] == M.shape[1] ### only square matrices
    assert M.dtype == np.float64 ### only high precision
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    Q = randomized_range_finder(M, n_random, n_iter, power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)
    gramB = safe_sparse_dot(B, B.T, dense_output=True)
    del B
    logging.info("Shape of computed gramian matrix is: %d x %d" % gramB.shape)
    s, Uhat = eigh(gramB)
    U = np.dot(Q, Uhat)
    return np.sqrt(s[-n_components:]), U[:, -n_components:]