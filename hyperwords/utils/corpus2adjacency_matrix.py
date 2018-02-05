import numpy as np
import time
from collections import Counter, defaultdict

from docopt import docopt
from scipy.sparse import csr_matrix, save_npz

from matrix_serializer import load_vocabulary


def main():
    args = docopt("""
    Usage:
        corpus2adjacency_matrix.py [options] <corpus> <vocab>

    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
    """)

    start = time.time()

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])

    print("Building vocabulary V")
    vocab_path = args["<vocab>"]
    if vocab_path:
        wi, _ = load_vocabulary(vocab_path)
    else:
        wi, _ = read_vocab(corpus_file, thr)
    n = len(wi)
    print("|V|=%d over threshold %d" % (n, thr))

    pair_counts = defaultdict(float)

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

    print("Building lists for adjacency matrix")
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

    print("Building adjacency matrix from lists of values and indices")
    adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)

    print("Writing the output")
    output_file_name = corpus_file + "_win=%d.adjacency" % win
    save_npz(output_file_name, adjacency_matrix)

    output_vocab_name = corpus_file + "_win=%d.words.vocab" % win

    with open(output_vocab_name, "w") as f:
        for word in sorted(wi, key=wi.get, reverse=False):
            f.write("%s\n" % word)

    print("Time elapsed: %d" % (time.time() - start))


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().split()))

        w2count = dict([(token, count) for token, count in vocab.items() if count >= thr])
    wi = {}

    for e, word in enumerate(sorted(w2count, key=w2count.get, reverse=True)):
        wi[word] = e
    return wi, w2count


if __name__ == '__main__':
    main()
