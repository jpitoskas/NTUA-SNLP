#!/bin/bash
source ./kaldi-master/egs/usc/path.sh
. ./kaldi-master/egs/usc/cmd.sh

cd ./kaldi-master/egs/usc/
export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin

build-lm.sh -i data/local/dict/lm_dev.text -n 1 -o data/local/lm_tmp/built_lm_dev_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 2 -o data/local/lm_tmp/built_lm_dev_bigram.ilm.gz

build-lm.sh -i data/local/dict/lm_test.text -n 1 -o data/local/lm_tmp/built_lm_test_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_test.text -n 2 -o data/local/lm_tmp/built_lm_test_bigram.ilm.gz

cd data/local/lm_tmp/

echo "Calculating perplexity for evaluation data unigram model"

prune-lm --threshold=1e-6,1e-6 built_lm_dev_unigram.ilm.gz built_lm_dev_unigram.plm
compile-lm built_lm_dev_unigram.plm --eval=../dict/lm_dev.text --dub=10000000

echo "Calculating perplexity for evaluation data bigram model"

prune-lm --threshold=1e-6,1e-6 built_lm_dev_bigram.ilm.gz built_lm_dev_bigram.plm
compile-lm built_lm_dev_bigram.plm --eval=../dict/lm_dev.text --dub=10000000


echo "Calculating perplexity for test data unigram model"

prune-lm --threshold=1e-6,1e-6 built_lm_test_unigram.ilm.gz built_lm_test_unigram.plm
compile-lm built_lm_test_unigram.plm --eval=../dict/lm_test.text --dub=10000000


echo "Calculating perplexity for test data bigram model"

prune-lm --threshold=1e-6,1e-6 built_lm_test_bigram.ilm.gz built_lm_test_bigram.plm
compile-lm built_lm_test_bigram.plm --eval=../dict/lm_test.text --dub=10000000
