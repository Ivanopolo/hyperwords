#!/usr/bin/env bash

#sleep 7000

export CORPUS=/home/ivan/hyperwords/data/svd_testing/wiki
pypy hyperwords/utils/corpus2adjacency_matrix.py --win 2 --thr 30 $CORPUS

export CORPUS=/home/ivan/hyperwords/data/svd_testing/wiki
pypy hyperwords/utils/corpus2adjacency_matrix.py --win 2 --thr 3 $CORPUS