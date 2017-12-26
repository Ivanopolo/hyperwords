from docopt import docopt
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
import os
import pickle
from functools import partial

from ..representations.matrix_serializer import load_vocabulary


class SpectralEvaluator(object):
    def __init__(self, input_path, deepwalk_window=5, deepwalk_negative_sampling=5.0):
        wi, iw = load_vocabulary(input_path + ".words.vocab")
        self.wi = wi
        self.iw = iw
        self.vecs = np.load(input_path + ".vecs.npy")
        print(self.vecs.shape, len(iw))
        self.degrees = np.load(input_path + ".degrees.npy")
        vals = np.load(input_path + ".vals.npy")

        commute_time_eigenscaling = np.sqrt(1.0 / vals)
        self.commute_time_vecs = commute_time_eigenscaling * self.vecs / np.expand_dims(np.sqrt(self.degrees), 1)

        deepwalk_eigenscaling = np.zeros(vals.shape[0])
        for i in range(1, deepwalk_window + 1):
            deepwalk_eigenscaling += vals**i

        deepwalk_eigenscaling = np.sqrt(deepwalk_eigenscaling / deepwalk_window)
        self.deepwalk_vecs = deepwalk_eigenscaling * self.vecs / np.expand_dims(np.sqrt(self.degrees), 1)

        self.C = np.sum(self.degrees) / deepwalk_negative_sampling

        self.dummy = np.zeros(vals.shape[0])
        self.eps = 1e-6

    def get_rep(self, word, vecs):
        if word in self.wi:
            return vecs[self.wi[word]]
        else:
            return self.dummy

    def _log_inner_product(self, x, y):
        return np.log(max(self.C * x.dot(y), self.eps))

    def log_cosine_similarity(self, x, y):
        inner_product = self._log_inner_product(x,y)
        xTx = self._log_inner_product(x,x)
        yTy = self._log_inner_product(y,y)
        return inner_product / np.sqrt(max(xTx * yTy, self.eps))

    def log_l2_similarity(self, x, y):
        inner_product = self._log_inner_product(x, y)
        xTx = self._log_inner_product(x, x)
        yTy = self._log_inner_product(y, y)
        return - (xTx + yTy - 2 * inner_product)

    def cosine_similarity(self, x, y):
        inner_product = x.dot(y)
        return inner_product / (max(norm(x) * norm(y), self.eps))

    def l2_similarity(self, x, y):
        return -np.mean((x - y)**2)

    def log_cosine_similarity_vecs(self, vocab_representation, vecs):
        inner_prods = np.log(np.maximum(self.C * vocab_representation.dot(vecs.T), self.eps))
        diagonal_prod = np.log(np.maximum(self.C * np.sum(vecs * vecs, axis=1), self.eps))
        self_prods = np.log(np.maximum(self.C * np.sum(vocab_representation * vocab_representation, axis=1, keepdims=True), self.eps))
        norms = np.maximum(np.tile(self_prods, [1, len(vecs)]) * np.tile(diagonal_prod, [len(self_prods), 1]), self.eps)
        cosines = inner_prods / np.sqrt(norms)
        return cosines

    def log_l2_similarity_vecs(self, vocab_representation, vecs):
        inner_prods = np.log(np.maximum(self.C * vocab_representation.dot(vecs.T), self.eps))
        diagonal_prod = np.log(np.maximum(self.C * np.sum(vecs * vecs, axis=1), self.eps))
        self_prods = np.log(np.maximum(self.C * np.sum(vocab_representation * vocab_representation, axis=1, keepdims=True), self.eps))
        l2 = -(-2 * inner_prods + np.tile(self_prods, [1, len(vecs)]) + np.tile(diagonal_prod, [len(self_prods), 1]))
        return l2

    def cosine_similarity_vecs(self, vocab_representation, vecs):
        normalized_vocab_repr = vocab_representation / np.maximum(norm(vocab_representation, axis=1, keepdims=True), self.eps)
        normalized_vecs = vecs / norm(vecs, axis=1, keepdims=True)
        return normalized_vocab_repr.dot(normalized_vecs.T)

    def l2_similarity_vecs(self, vocab_representation, vecs):
        inner_prods = vocab_representation.dot(vecs.T)
        diagonal_prod = np.sum(vecs * vecs, axis=1)
        self_prods = np.sum(vocab_representation * vocab_representation, axis=1, keepdims=True)
        return - (-2*inner_prods + np.tile(diagonal_prod, [len(self_prods), 1]) + np.tile(self_prods, [1, len(vecs)]))


def main():
    args = docopt("""
    Usage:
        eval_all.py <spectral_embeddings_path> <ws_datasets_dir> <analogy_datasets_dir>
    """)

    input_path = args["<spectral_embeddings_path>"]

    embs = SpectralEvaluator(input_path)

    vecs_list = {
        "Unscaled": partial(embs.get_rep, vecs=embs.vecs),
        "CommuteTime": partial(embs.get_rep, vecs=embs.commute_time_vecs),
        "Deepwalk": partial(embs.get_rep, vecs=embs.deepwalk_vecs)
    }

    sim_fun_list = {
        "Cos": embs.cosine_similarity,
        "L2": embs.l2_similarity,
        "LogCos": embs.log_cosine_similarity,
        "LogL2": embs.log_l2_similarity
    }

    ws_datasets_dir = args["<ws_datasets_dir>"]
    ws_datasets = os.listdir(ws_datasets_dir)

    results = {}

    for dataset in ws_datasets:
        data = read_ws_test_set(os.path.join(ws_datasets_dir, dataset))

        for sim_name, sim_fun in sim_fun_list.items():
            for get_name, gen_fun in vecs_list.items():
                correlation = evaluate_ws(sim_fun, gen_fun, data)
                res_name = "_".join([dataset, sim_name, get_name])
                results[res_name] = correlation
                print(res_name, correlation)

    vecs_list = {
        "Unscaled": embs.vecs,
        "CommuteTime": embs.commute_time_vecs,
        "Deepwalk": embs.deepwalk_vecs
    }

    sim_fun_list = {
        "Cos": embs.cosine_similarity_vecs,
        "L2": embs.l2_similarity_vecs,
        "LogCos": embs.log_cosine_similarity_vecs,
        "LogL2": embs.log_l2_similarity_vecs
    }

    analogy_datasets_dir = args["<analogy_datasets_dir>"]
    analogy_datasets = os.listdir(analogy_datasets_dir)

    for dataset in analogy_datasets:
        data = read_analogy_test_set(os.path.join(analogy_datasets_dir, dataset))
        xi, ix = get_vocab(data)

        for sim_name, sim_fun in sim_fun_list.items():
            for get_name, vecs in vecs_list.items():
                sims = prepare_similarities(vecs, embs.wi, ix, sim_fun)
                accuracy_add, accuracy_mul = evaluate_analogy(sims, embs.wi, embs.iw, xi, data)
                res_name = "_".join([dataset, sim_name, get_name])
                results[res_name] = (accuracy_add, accuracy_mul)
                print(res_name, accuracy_add, accuracy_mul)

    pickle.dump(results, open(input_path + ".res", "wb"))


def read_ws_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            test.append(((x, y), float(sim)))
    return test


def evaluate_ws(sim_fun, get_fun, data):
    results = []
    for (x_id, y_id), sim in data:
        x = get_fun(x_id)
        y = get_fun(y_id)
        results.append((sim_fun(x, y), sim))
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0]


def read_analogy_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            analogy = line.strip().lower().split()
            test.append(analogy)
    return test


def get_vocab(data):
    vocab = set()
    for analogy in data:
        vocab.update(analogy)
    vocab = sorted(vocab)
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def prepare_similarities(vecs, wi, vocab, sim_fun):
    vocab_representation = vecs[[wi[w] if w in wi else 0 for w in vocab]]

    for i, w in enumerate(vocab):
        if w not in wi:
            vocab_representation[i] = np.zeros(vocab_representation.shape[1])

    sims = sim_fun(vocab_representation, vecs)
    sims = (sims+1) / 2
    # sims -= np.min(sims)
    # sims /= np.max(sims)
    return sims


def evaluate_analogy(sims, wi, iw, xi, data):
    correct_add = 0.0
    correct_mul = 0.0
    for i, (a, a_, b, b_) in enumerate(data):
        b_add, b_mul = guess(sims, wi, iw, xi, a, a_, b)
        if b_add == b_:
            correct_add += 1
        if b_mul == b_:
            correct_mul += 1

    return correct_add/len(data), correct_mul/len(data)


def guess(sims, wi, iw, xi, a, a_, b):
    sa = sims[xi[a]]
    sa_ = sims[xi[a_]]
    sb = sims[xi[b]]

    add_sim = -sa + sa_ + sb
    if a in wi: add_sim[wi[a]] = 0
    if a_ in wi: add_sim[wi[a_]] = 0
    if b in wi: add_sim[wi[b]] = 0
    b_add = iw[np.argmax(add_sim)]

    mul_sim = sa_ * sb * np.reciprocal(sa + 0.01)
    if a in wi: mul_sim[wi[a]] = 0
    if a_ in wi: mul_sim[wi[a_]] = 0
    if b in wi: mul_sim[wi[b]] = 0
    b_mul = iw[np.argmax(mul_sim)]

    return b_add, b_mul


if __name__ == '__main__':
    main()
