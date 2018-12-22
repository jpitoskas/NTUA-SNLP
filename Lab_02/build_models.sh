#!/usr/bin/env bash
source ./kaldi-master/egs/wsj/s5/path.sh
# dir = $pwd/kaldi-master/egs/usc/data/local
# langdir = $pwd/kaldi-master/egs/usc/data/lang

build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 1 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_unigram.ilm.gz
build-lm.sh -i ./kaldi-master/egs/usc/data/local/dict/lm_train.text -n 2 -o ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_bigram.ilm.gz

compile-lm ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./kaldi-master/egs/usc/data/local/nist_lm/compiled_lm_unigram.arpa.gz
compile-lm ./kaldi-master/egs/usc/data/local/lm_tmp/built_lm_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./kaldi-master/egs/usc/data/local/nist_lm/compiled_lm_bigram.arpa.gz

cd ./kaldi-master/egs/wsj/s5
utils/prepare_lang.sh ../../usc/data/local/dict '!sil' ../../usc/data/lang ../../usc/data/lang
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
