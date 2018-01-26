import time

from docopt import docopt
from scipy.sparse import load_npz

from ..utils.tools import estimate_rhoB


def main():
    args = docopt("""
    Usage:
        find_rhoB.py [options] <adjacency_matrix_path>
    """)

    '''
    Using SLP algorithm to estimate second eigenvalue of the non-backtracking operator
    See for reference: https://arxiv.org/pdf/1406.1880.pdf
    And MATLAB implementation: http://mode_net.krzakala.org/
    '''

    start = time.time()
    print("Loading adjacency matrix, %f" % time.time())
    adjacency_matrix_path = args["<adjacency_matrix_path>"]
    adjacency_matrix = load_npz(adjacency_matrix_path + ".adjacency.npz")
    adjacency_matrix.data = adjacency_matrix.data

    _ = estimate_rhoB(adjacency_matrix)

    print("Time elapsed %f" % (time.time() - start))


if __name__ == '__main__':
    main()