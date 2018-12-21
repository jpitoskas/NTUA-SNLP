#!/usr/bin/env bash
source ./kaldi-master/egs/wsj/s5/path.sh

build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 1 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_unigram.ilm.gz
build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 2 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_bigram.ilm.gz
