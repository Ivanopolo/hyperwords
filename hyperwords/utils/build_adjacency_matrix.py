from scipy.sparse import csr_matrix, save_npz
import numpy as np
from docopt import docopt


def main():
    args = docopt("""
        Usage:
            build_adjacency_matrix.py <vocab_path> <counts_path> <output_path>
        """)

    print("Building the word2id index..")
    vocab_path = args["<vocab_path>"]
    word2id = {}
    with open(vocab_path, 'r') as f:
        for idx, line in enumerate(f):
            split_line = line.strip().split(",")
            word = ",".join(split_line[:len(split_line) - 1])
            word2id[word] = idx

    print("Reading the count data...")
    N = len(word2id)
    data = []
    rows = []
    cols = []

    with open(args["<counts_path>"], 'r') as f:
        for line in f:
            count, word_a, word_b = line.strip().split()
            word_a_id = word2id[word_a]
            word_b_id = word2id[word_b]

            data.append(float(count))
            rows.append(word_a_id)
            cols.append(word_b_id)

    print("Building the adjacency matrix...")
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=[N, N], dtype=np.float64)
    save_npz(args["<output_path>"], adjacency_matrix)


if __name__ == '__main__':
    main()