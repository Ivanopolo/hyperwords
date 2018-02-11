#!/usr/bin/env bash
### Timing experiments on SVD with different dimensions
export COUNTS=/home/ivan/hyperwords/data/svd_testing/wiki_win=2_thr=30

export OPTS="--dim 100 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd $OPTS $COUNTS

export OPTS="--pow 0.3 --dim 100 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

export COUNTS=/home/ivan/hyperwords/data/svd_testing/wiki_win=2_thr=3

export OPTS="--dim 100 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd $OPTS $COUNTS

export OPTS="--pow 0.3 --dim 100 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

