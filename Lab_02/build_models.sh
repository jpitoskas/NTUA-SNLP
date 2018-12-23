#!/bin/bash
source ./kaldi-master/egs/wsj/s5/path.sh
# dir = $pwd/kaldi-master/egs/usc/data/local
# langdir = $pwd/kaldi-master/egs/usc/data/lang

# python3 omeleta.py

cd ./kaldi-master/egs/usc/
build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/built_lm_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/built_lm_bigram.ilm.gz

compile-lm data/local/lm_tmp/built_lm_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/compiled_lm_unigram.arpa.gz
compile-lm data/local/lm_tmp/built_lm_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/compiled_lm_bigram.arpa.gz

# utils/validate_dict_dir.pl lang/dict
sudo utils/prepare_lang.sh data/local/dict '<oov>' data/lang data/lang
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib


for x in train dev test; do
  utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
done

echo Created spk2utt files

echo Preparing language models for test

lmdir=$PWD/data/local/nist_lm
tmpdir=$PWD/data/local/lm_tmp
lexicon=$PWD/data/local/dict/lexicon.txt

for lm_suffix in unigram bigram; do
  test=$PWD/data/lang_test_${lm_suffix}
  mkdir -p $test
  cp -r $PWD/data/lang/* $test

  gunzip -c $lmdir/compiled_lm_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst
  fstisstochastic $test/G.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

 # Everything below is only for diagnostic.
 # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
 # this might cause determinization failure of CLG.
 # #0 is treated as an empty word.
  mkdir -p $tmpdir/g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
  fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
   fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
  fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' &&
    echo "Language model has cycles with empty words" && exit 1
  rm -r $tmpdir/g
done

echo "Succeeded in formatting data."
