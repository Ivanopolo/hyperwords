### Timing experiments on SVD with different dimensions
export COUNTS=/home/ivan/data/svd_testing/wiki_win=2_thr=100

export OPTS="--dim 50 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd --dim 1 --neg 1 --cds 0.75 $COUNTS

export OPTS="--dim 100 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd --dim 1 --neg 1 --cds 0.75 $COUNTS

export OPTS="--dim 300 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd --dim 1 --neg 1 --cds 0.75 $COUNTS

export OPTS="--dim 500 --neg 1 --cds 0.75"
python -m hyperwords.utils.counts2svd --dim 1 --neg 1 --cds 0.75 $COUNTS