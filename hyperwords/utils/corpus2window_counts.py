from collections import Counter, defaultdict
import pickle

from docopt import docopt


def main():
    args = docopt("""
    Usage:
        corpus2window_counts.py [options] <corpus>

    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])

    vocab = read_vocab(corpus_file, thr)
    window_counts = defaultdict(int)

    with open(corpus_file) as f:
        for line in f:
            tokens = [t if t in vocab else None for t in line.strip().split()]
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
                            window = abs(i - j)
                            if tok < tokens[j]:
                                key_pair = (tok, tokens[j], window)
                            else:
                                key_pair = (tokens[j], tok, window)

                            window_counts[key_pair] += 1

    pickle.dump(window_counts, open(corpus_file + ".window_counts", "wb"))


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().split()))
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
