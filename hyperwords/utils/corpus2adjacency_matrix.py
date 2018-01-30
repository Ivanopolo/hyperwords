import numpy as np
from collections import Counter, defaultdict

from docopt import docopt
from scipy.sparse import csr_matrix, save_npz


def main():
    args = docopt("""
    Usage:
        corpus2adjacency_matrix.py [options] <corpus>

    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])

    wi = read_vocab(corpus_file, thr)
    n = len(wi)
    pair_counts = defaultdict(int)

    with open(corpus_file) as f:
        for e, line in enumerate(f):
            if e % 10 ** 6 == 0:
                print("Line ", e)

            tokens = [wi[t] if t in wi else None for t in line.strip().split()]
            len_tokens = len(tokens)

            for i, tok in enumerate(tokens):
                if tok is not None:
                    start = i - win
                    if start < 0:
                        start = 0
                    end = i + win + 1
                    if end > len_tokens:
                        end = len_tokens

                    for j in range(start, end):
                        if j != i and tokens[j] is not None:
                            if tok < tokens[j]:
                                key_pair = (tok, tokens[j])
                            else:
                                key_pair = (tokens[j], tok)

                                pair_counts[key_pair] += 1

    data = []
    row_inds = []
    col_inds = []

    for (idx_a, idx_b), value in pair_counts.items():
        row_inds.append(idx_a)
        col_inds.append(idx_b)
        data.append(value)

        row_inds.append(idx_b)
        col_inds.append(idx_a)
        data.append(value)

    adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
    output_file_name = corpus_file + "win=%d.adjacency" % win
    save_npz(output_file_name, adjacency_matrix)


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().split()))

    vocab = dict([(token, count) for token, count in vocab.items() if count >= thr])
    wi = {}

    for e, (word, count) in enumerate(sorted(vocab.items(), key=lambda k, v: (v, k))):
        wi[word] = e
        print(word, count)
    return wi


if __name__ == '__main__':
    main()
