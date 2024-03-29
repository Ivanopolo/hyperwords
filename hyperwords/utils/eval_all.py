from docopt import docopt
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
import os
import pickle
from functools import partial
from sklearn import mixture
from sklearn.utils import resample

from ..representations.matrix_serializer import load_vocabulary


class SpectralEvaluator(object):
    def __init__(self, input_path):
        wi, iw = load_vocabulary(input_path + ".words.vocab")
        self.wi = wi
        self.iw = iw
        self.vecs = np.load(input_path + ".vecs.npy")
        vals = np.abs(np.load(input_path + ".vals.npy"))

        commute_time_eigenscaling = np.sqrt(1.0 / vals)
        self.commute_time_vecs = commute_time_eigenscaling * self.vecs
        self.sqrt_vecs = np.sqrt(vals) * self.vecs
        self.eps = 1e-6

    def get_rep(self, word, vecs):
        if word in self.wi:
            return vecs[self.wi[word]]
        else:
            raise IndexError("There is not word %s in the dictionary" % word)

    def cosine_similarity(self, x, y):
        inner_product = x.dot(y)
        return inner_product / (max(norm(x) * norm(y), self.eps))

    @staticmethod
    def inner_product(x, y):
        return x.dot(y)

    def cosine_similarity_vecs(self, vocab_representation, vecs):
        normalized_vocab_repr = vocab_representation / np.maximum(norm(vocab_representation, axis=1, keepdims=True), self.eps)
        normalized_vecs = vecs / norm(vecs, axis=1, keepdims=True)
        return normalized_vocab_repr.dot(normalized_vecs.T)


def compute_confidence_intervals(results):
    gmm = mixture.GaussianMixture(n_components=1, covariance_type='spherical')
    gmm.fit(np.array(results).reshape(-1, 1))
    mean = gmm.means_[0, 0]
    ci_95 = 1.96 * np.sqrt(gmm.covariances_[0])
    return mean, ci_95


def main():
    args = docopt("""
    Usage:
        eval_all.py [options] <spectral_embeddings_path> <ws_datasets_dir> <analogy_datasets_dir>
        
        Options:
        --with_ci              Output confidence interval (takes way longer to evaluate)
        --n_bootstraps NUM     Number of bootstraps [default: 100]
    """)

    input_path = args["<spectral_embeddings_path>"]

    embs = SpectralEvaluator(input_path)

    vecs_list = {
        "Unscaled": partial(embs.get_rep, vecs=embs.vecs),
        #"CommuteTime": partial(embs.get_rep, vecs=embs.commute_time_vecs),
        "Sqrt": partial(embs.get_rep, vecs=embs.sqrt_vecs)
    }

    sim_fun_list = {
        "Cos": embs.cosine_similarity,
        "IP": embs.inner_product
    }

    ws_datasets_dir = args["<ws_datasets_dir>"]
    ws_datasets = os.listdir(ws_datasets_dir)

    results = {}

    for dataset in ws_datasets:
        data = read_ws_test_set(os.path.join(ws_datasets_dir, dataset), embs.wi)

        for sim_name, sim_fun in sim_fun_list.items():
            for get_name, gen_fun in vecs_list.items():
                if args["--with_ci"]:
                    n_boots = int(args["--n_bootstraps"])
                    resampled_correlation = []

                    for _ in range(n_boots):
                        re_sampled_data = resample(data)
                        correlation = evaluate_ws(sim_fun, gen_fun, re_sampled_data)
                        resampled_correlation.append(correlation)

                    correlation = compute_confidence_intervals(resampled_correlation)
                else:
                    correlation = evaluate_ws(sim_fun, gen_fun, data)
                res_name = "_".join([dataset, sim_name, get_name])
                results[res_name] = correlation
                print(res_name, correlation)

    vecs_list = {
        "Unscaled": embs.vecs,
        #"CommuteTime": embs.commute_time_vecs,
        #"Sqrt": embs.sqrt_vecs
    }

    sim_fun_list = {
        "Cos": embs.cosine_similarity_vecs
    }

    analogy_datasets_dir = args["<analogy_datasets_dir>"]
    analogy_datasets = os.listdir(analogy_datasets_dir)

    for dataset in analogy_datasets:
        data = read_analogy_test_set(os.path.join(analogy_datasets_dir, dataset), embs.wi)
        xi, ix = get_vocab(data)

        for sim_name, sim_fun in sim_fun_list.items():
            for get_name, vecs in vecs_list.items():
                sims = prepare_similarities(vecs, embs.wi, ix, sim_fun)

                if args["--with_ci"]:
                    n_boots = int(args["--n_bootstraps"]) // 3
                    resampled_accuracy_add = []
                    resampled_accuracy_mul = []

                    for i in range(n_boots):
                        re_sampled_data = resample(data)
                        accuracy_add, accuracy_mul = evaluate_analogy(sims, embs.wi, embs.iw, xi, re_sampled_data)
                        resampled_accuracy_add.append(accuracy_add)
                        resampled_accuracy_mul.append(accuracy_mul)

                    accuracy_add = compute_confidence_intervals(resampled_accuracy_add)
                    accuracy_mul = compute_confidence_intervals(resampled_accuracy_mul)
                else:
                    accuracy_add, accuracy_mul = evaluate_analogy(sims, embs.wi, embs.iw, xi, data)

                res_name = "_".join([dataset, sim_name, get_name])
                results[res_name] = (accuracy_add, accuracy_mul)
                print(res_name, accuracy_add, accuracy_mul)

    if args["--with_ci"]:
        pickle.dump(results, open(input_path + "_with_ci.res", "wb"))
    else:
        pickle.dump(results, open(input_path + ".res", "wb"))


def read_ws_test_set(path, wi):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            if x in wi and y in wi:
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


def read_analogy_test_set(path, wi):
    test = []
    with open(path) as f:
        for line in f:
            analogy = line.strip().lower().split()
            found_words = [w for w in analogy if w in wi]
            if len(found_words) == len(analogy):
                test.append(analogy)
    return test


def get_vocab(data):
    vocab = set()
    for analogy in data:
        vocab.update(analogy)
    vocab = sorted(vocab)
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def get_vec(vecs, word, wi):
    if word in wi:
        return vecs[wi[word]]
    else:
        return np.zeros(vecs.shape[1])


def prepare_new_vecs(vecs, data, wi):
    query_vecs = np.zeros([len(data), vecs.shape[1]], dtype=np.float64)
    expected_result = -np.ones(len(data))

    for i, (a, a_, b, b_) in enumerate(data):
        va = get_vec(vecs, a, wi)
        va_ = get_vec(vecs, a_, wi)
        vb = get_vec(vecs, b, wi)
        query_vec = -va + va_ + vb
        query_vecs[i] = query_vec
        if b_ in wi:
            expected_result[i] = wi[b_]

    return query_vecs, expected_result


def evaluate_analogy2(sims, data, expected_results, wi):
    for i, (a, a_, b, b_) in enumerate(data):
        if a in wi: sims[i][wi[a]] = 0
        if a_ in wi: sims[i][wi[a_]] = 0
        if b in wi: sims[i][wi[b]] = 0

    answers = np.argmax(sims, axis=1)
    score = (answers == expected_results).sum()
    return score


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
