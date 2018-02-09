### Timing experiments on already established solution with optimized SLEPc installation
export COUNTS=/home/ivan/data/svd_testing/wiki_win=2_thr=100

export OPTS="--pow 0.3 --dim 50 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

export OPTS="--pow 0.3 --dim 100 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

export OPTS="--pow 0.3 --dim 300 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian

export OPTS="--pow 0.3 --dim 500 --max_iter 100 --tol 0.01"
python -m hyperwords.utils.learn_spectral_embeddings $OPTS $COUNTS bethe_hessian