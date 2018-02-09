from scipy.sparse import csr_matrix, dok_matrix
from sparsesvd import sparsesvd

from docopt import docopt
import numpy as np
import time

from ..representations.explicit import PositiveExplicitLoaded
from ..representations.matrix_serializer import save_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2svd.py [options] <counts_path> <output_path>

    Options:
        --dim NUM    Dimensionality of eigenvectors [default: 500]
        --neg NUM    Number of negative samples; subtracts its log from PMI [default: 1]
        --cds NUM    Context distribution smoothing [default: 0.75]
    """)

    counts_path = args['<counts_path>']
    output_path = args['<output_path>']
    dim = int(args['--dim'])
    neg = int(args['--neg'])
    cds = float(args['--cds'])

    data = np.load(counts_path + ".data.npz")
    row_inds = np.load(counts_path + ".row_inds.npz")
    col_inds = np.load(counts_path + ".col_inds.npz")
    adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), dtype=np.float64)

    pmi = build_pmi_matrix(adjacency_matrix, cds)
    explicit = PositiveExplicitLoaded(counts_path, pmi, normalize=False, neg=neg)

    start = time.time()
    ut, s, vt = sparsesvd(explicit.m.tocsc(), dim)
    print("Time elapsed for SVD: %f" % (time.time() - start))

    np.save(output_path + '.vecs.npy', ut.T)
    np.save(output_path + '.vals.npy', s)
    save_vocabulary(output_path + '.words.vocab', explicit.iw)


def build_pmi_matrix(adjacency_matrix, cds):
    sum_w = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    sum_c = sum_w.copy()
    sum_c = sum_c ** cds

    sum_total = sum_w.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = multiply_by_rows(adjacency_matrix, sum_w)
    pmi = multiply_by_columns(adjacency_matrix, sum_c)
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


if __name__ == '__main__':
    main()
