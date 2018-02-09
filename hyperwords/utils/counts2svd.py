from scipy.sparse import csr_matrix, dok_matrix
from sparsesvd import sparsesvd

from docopt import docopt
import numpy as np
import time

from ..representations.matrix_serializer import save_vocabulary, load_vocabulary


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

    _, iw = load_vocabulary(counts_path + '.words.vocab')
    adjacency_matrix = load_adjacency_matrix(counts_path)
    ppmi = build_ppmi_matrix(adjacency_matrix, cds, neg)

    start = time.time()
    ut, s, vt = sparsesvd(ppmi.tocsc(), dim)
    print("Time elapsed for SVD: %f" % (time.time() - start))

    np.save(output_path + '.vecs.npy', ut.T)
    np.save(output_path + '.vals.npy', s)
    save_vocabulary(output_path + '.words.vocab', iw)


def load_adjacency_matrix(counts_path):
    data = np.load(counts_path + ".data.npz")["arr_0"]
    row_inds = np.load(counts_path + ".row_inds.npz")["arr_0"]
    col_inds = np.load(counts_path + ".col_inds.npz")["arr_0"]
    adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), dtype=np.float64)
    return adjacency_matrix


def build_ppmi_matrix(adjacency_matrix, neg, cds):
    sum_w = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    sum_c = sum_w.copy()
    sum_c = sum_c ** cds

    sum_total = sum_w.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = multiply_by_rows(adjacency_matrix, sum_w)
    pmi = multiply_by_columns(adjacency_matrix, sum_c)
    pmi = pmi * sum_total

    pmi.data = np.log(pmi.data)

    pmi.data -= np.log(neg)
    pmi.data[pmi.data < 0] = 0
    pmi.eliminate_zeros()

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
