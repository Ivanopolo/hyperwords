import numpy as np
import time
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

    start_time = time.time()

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])

    print("Building vocabulary V")
    wi, _ = read_vocab(corpus_file, thr)
    n = len(wi)
    print("|V|=%d over threshold %d" % (n, thr))

    pair_counts = defaultdict(float)

    with open(corpus_file) as f:
        for e, line in enumerate(f):
            if e % 10 ** 6 == 0:
                print("Line ", e)

            tokens = [wi[t] for t in line.strip().split() if t in wi]
            len_tokens = len(tokens)

            for i, tok in enumerate(tokens):
                end = i + win + 1
                if end > len_tokens:
                    end = len_tokens

                for j in range(i+1, end):
                    if tok < tokens[j]:
                        key_pair = (tok, tokens[j])
                    else:
                        key_pair = (tokens[j], tok)

                    pair_counts[key_pair] += 1

    print("Building lists for adjacency matrix")
    data = np.zeros(len(pair_counts) * 2, dtype=np.float32)
    row_inds = np.zeros(len(pair_counts) * 2, dtype=np.int32)
    col_inds = np.zeros(len(pair_counts) * 2, dtype=np.int32)
    num_vals = len(pair_counts)

    for i, ((idx_a, idx_b), value) in enumerate(pair_counts.items()):
        row_inds[i] = idx_a
        col_inds[i] = idx_b
        data[i] = value

        row_inds[i + num_vals] = idx_b
        col_inds[i + num_vals] = idx_a
        data[i + num_vals] = value

    adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), dtype=np.float64)
    output_file_name = corpus_file + "_win=%d_thr=%d" % (win, thr)
    save_npz(output_file_name + ".adjacency", adjacency_matrix)
    output_vocab_name = corpus_file + "_win=%d_thr=%d.words.vocab" % (win, thr)

    with open(output_vocab_name, "w") as f:
        for word in sorted(wi, key=wi.get, reverse=False):
            f.write("%s\n" % word)

    print("Time elapsed: %d" % (time.time() - start_time))


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) > 1:
                vocab.update(Counter(split_line))

        w2count = dict([(token, count) for token, count in vocab.items() if count >= thr])
    wi = {}

    for e, word in enumerate(sorted(w2count, key=w2count.get, reverse=True)):
        wi[word] = e

    return wi, w2count


if __name__ == '__main__':
    main()
