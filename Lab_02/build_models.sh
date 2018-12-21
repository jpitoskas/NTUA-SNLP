#!/bin/bash
source ./kaldi-master/egs/wsj/s5/path.sh

python3 omeleta.py

build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 1 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_unigram.ilm.gz
build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 2 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_bigram.ilm.gz

compile-lm ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./kaldi-master/egs/usc/data/local/nist_lm/compiled_lm_unigram.arpa.gz
compile-lm ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./kaldi-master/egs/usc/data/local/nist_lm/compiled_lm_bigram.arpa.gz

utils/prepare_lang.sh ./kaldi-master/egs/usc/data/local/dict 'sil' ./kaldi-master/egs/usc/data/local/lang ./kaldi-master/egs/usc/data/lang
