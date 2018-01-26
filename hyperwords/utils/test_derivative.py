import numpy as np
import scipy.sparse
from scipy.sparse.linalg import norm
from docopt import docopt

from ..utils.tools import build_weighted_bethe_hessian, build_weighted_bethe_hessian_derivative, \
    build_weighted_bethe_hessian_derivative_direct, build_weighted_bethe_hessian_direct


def main():
    args = docopt("""
    Usage:
        test_derivative.py <adjacency_matrix_path>
    """)

    print("Loading the adjacency matrix")
    adjacency_matrix_path = args["<adjacency_matrix_path>"]
    adjacency_matrix = scipy.sparse.load_npz(adjacency_matrix_path + ".adjacency.npz")
    adjacency_matrix.data = adjacency_matrix.data ** 0.0
    adjacency_matrix = adjacency_matrix[:5,:]
    adjacency_matrix = adjacency_matrix[:, :5]
    #adjacency_matrix = adjacency_matrix.todense()
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()

    print("Computing the bethe hessians")
    rhoB = (degrees ** 2).mean() / degrees.mean() - 1
    print("Base rhoB is ", rhoB)
    BH = build_weighted_bethe_hessian(adjacency_matrix, rhoB)
    BHprime = build_weighted_bethe_hessian_derivative_direct(adjacency_matrix, rhoB)

    print("Starting the test")
    for eps in [1e-10, 1e-5]:
        true_BH = build_weighted_bethe_hessian(adjacency_matrix, rhoB + eps)
        finite_diff_gradient_matrix = (true_BH - BH) / eps
        print("EPS", eps)
        print(finite_diff_gradient_matrix)
        print(BHprime)


if __name__ == '__main__':
    main()