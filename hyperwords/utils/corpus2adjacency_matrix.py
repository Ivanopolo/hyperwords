import numpy as np
import time
from collections import Counter, defaultdict

from docopt import docopt


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

    # print("Building adjacency matrix from lists of values and indices")
    # adjacency_matrix = csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
    #
    # print("Writing the output")
    # output_file_name = corpus_file + "_win=%d.adjacency" % win
    # save_npz(output_file_name, adjacency_matrix)
    output_file_name = corpus_file + "_win=%d" % win
    np.savez(output_file_name + ".data", np.array(data))
    np.savez(output_file_name + ".row_inds", np.array(row_inds))
    np.savez(output_file_name + ".col_inds", np.array(col_inds))

    output_vocab_name = corpus_file + "_win=%d.words.vocab" % win

    with open(output_vocab_name, "w") as f:
        for word in sorted(wi, key=wi.get, reverse=False):
            f.write("%s\n" % word)

    print("Time elapsed: %d" % (time.time() - start_time))


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
