#!/usr/bin/env bash
### Timing experiments on SVD with different dimensions
export COUNTS=/Users/i.lobov/hyperwords/data/wiki/wikipedia.corpus.nodups_counts_win=1

export OPTS="--dim 100 --neg 1 --pos 100 --cds 0.75"
python -m hyperwords.utils.counts2svd $OPTS $COUNTS

EMBS=/Users/i.lobov/hyperwords/data/wiki/wikipedia.corpus.nodups_counts_win=1_svd_dim=100_neg=1_pos=100_cds=0.75
WS=/Users/i.lobov/hyperwords/testsets/ws
ANAL=/Users/i.lobov/hyperwords/testsets/analogy
python -m hyperwords.utils.eval_all $EMBS $WS $ANAL

export OPTS="--pow 0.3 --dim 100 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

export COUNTS=/home/ivan/hyperwords/data/svd_testing/wiki_win=2_thr=3

export OPTS="--dim 100 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd $OPTS $COUNTS

export OPTS="--pow 0.3 --dim 100 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

