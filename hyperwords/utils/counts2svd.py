import logging
import time
from sparsesvd import sparsesvd

import numpy as np
import os
from docopt import docopt
from scipy.sparse import csr_matrix, dok_matrix, load_npz

from ..representations.matrix_serializer import save_vocabulary, load_vocabulary
from ..utils.randomized import randomized_eigh, normalized_embedder


def main():
    args = docopt("""
    Usage:
        counts2svd.py [options] <counts_path>

    Options:
        --dim NUM           Dimensionality of eigenvectors [default: 500]
        --neg NUM           Number of negative samples; subtracts its log from PMI [default: 1]
        --pos NUM           Number of positive samples; add its log to PMI [default: 1]
        --cds NUM           Context distribution smoothing [default: 0.75]
        --randomized        Use randomized SVD
        --normalized        Use normalized embedder
        --oversample NUM    Number of oversamples in randomized SVD [default: 10]
        --power_iter NUM    Number of iterations of power method in randomized SVD [default: 2]
    """)

    start = time.time()

    counts_path = args['<counts_path>']
    dim = int(args['--dim'])
    neg = int(args['--neg'])
    pos = int(args['--pos'])
    cds = float(args['--cds'])
    randomized = args['--randomized']
    normalized = args['--normalized']
    oversample = int(args['--oversample'])
    power_iter = int(args['--power_iter'])

    output_path = counts_path + "_svd_dim=%d_neg=%d_pos=%d_cds=%.2f" % (dim, neg, pos, cds)
    if randomized:
        output_path += "_rand_oversample=%d_power_iter=%d" % (oversample, power_iter)

    logging.basicConfig(filename=output_path + ".log", filemode="w", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    _, iw = load_vocabulary(counts_path + '.words.vocab')
    adjacency_matrix = load_adjacency_matrix(counts_path)
    ppmi = build_ppmi_matrix(adjacency_matrix, cds, neg, pos)

    start_learning = time.time()
    logging.info("Starting SVD")
    if randomized:
        s, ut = randomized_eigh(ppmi, dim, oversample, power_iter)
    elif normalized:
        ut = normalized_embedder(ppmi, dim, power_iter)
        s = np.zeros(dim)
    else:
        ut, s, _ = sparsesvd(ppmi.tocsc(), dim)
        ut = ut.T

    logging.info("Time elapsed on learning: %f" % (time.time() - start_learning))

    np.save(output_path + '.vecs.npy', ut)
    np.save(output_path + '.vals.npy', s)
    save_vocabulary(output_path + '.words.vocab', iw)
    logging.info("Time elapsed: %f" % (time.time() - start))


def load_adjacency_matrix(counts_path):
    if os.path.exists(counts_path + ".adjacency.npz"):
        adjacency_matrix = load_npz(counts_path + ".adjacency.npz")
    else:
        data = np.load(counts_path + ".data.npz")["arr_0"]
        row_inds = np.load(counts_path + ".row_inds.npz")["arr_0"]
        col_inds = np.load(counts_path + ".col_inds.npz")["arr_0"]
        adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), dtype=np.float64)
    return adjacency_matrix


def build_ppmi_matrix(adjacency_matrix, cds, neg, pos):
    sum_w = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    sum_c = sum_w.copy()
    sum_c = sum_c ** cds

    sum_total = sum_w.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = multiply_by_rows(adjacency_matrix, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total

    pmi.data = np.log(pmi.data)

    pmi.data = pmi.data - np.log(neg) + np.log(pos)
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
