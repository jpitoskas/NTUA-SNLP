#!/bin/bash
source ./kaldi-master/egs/usc/path.sh
cd ./kaldi-master/egs/usc/

# Train Monophone
steps/train_mono.sh --nj 4 data/train data/lang exp/mono || exit 1;
# steps/train_lda_mllt.sh

# Generate graphs for unigram and bigram
utils/mkgraph.sh data/lang_test_unigram exp/mono exp/mono/unigram_graph || exit 1;
utils/mkgraph.sh data/lang_test_bigram exp/mono exp/mono/bigram_graph || exit 1;

# fstdraw --isymbols=exp/mono/unigram_graph/words.txt --osymbols=exp/mono/unigram_graph/words.txt -portrait exp/mono/unigram_graph/HCLG.fst | dot -Gdpi=400 -Tjpg > exp/mono/unigram_graph/mono_unigram.jpg
# fstdraw --isymbols=exp/mono/bigram_graph/words.txt --osymbols=exp/mono/bigram_graph/words.txt -portrait exp/mono/bigram_graph/HCLG.fst | dot -Gdpi=400 -Tjpg > exp/mono/bigram_graph/mono_bigram.jpg

# Decode
steps/decode.sh --nj 4 exp/mono/unigram_graph data/dev exp/mono/decode_uni_dev
steps/decode.sh --nj 4 exp/mono/unigram_graph data/test exp/mono/decode_uni_test

# Alignment
steps/align_si.sh --nj 4 data/train data/lang_test_unigram exp/mono exp/mono_ali

# Train triphone
steps/train_deltas.sh 2000 10000 data/train data/lang_test_unigram exp/mono_ali exp/tri

utils/mkgraph.sh data/lang_test_unigram exp/tri exp/tri/graph
steps/decode.sh --nj 4 exp/tri/graph data/test exp/tri/decode
