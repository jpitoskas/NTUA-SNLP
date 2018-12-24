#!/bin/bash
source ./kaldi-master/egs/usc/path.sh
cd ./kaldi-master/egs/usc/

# Train Monophone
steps/train_mono.sh --nj 4 data/train data/lang exp/mono || exit 1;
# steps/train_lda_mllt.sh

# Generate graphs for unigram and bigram
utils/mkgraph.sh data/lang_test_unigram exp/mono exp/mono/unigram_graph || exit 1;
utils/mkgraph.sh data/lang_test_bigram exp/mono exp/mono/bigram_graph || exit 1;

fstdraw -portrait exp/mono/unigram_graph/HCLG.fst | dot -Gdpi=400 -Tjpg > exp/mono/unigram_graph/mono_unigram.jpg
fstdraw -portrait exp/mono/bigram_graph/HCLG.fst | dot -Gdpi=400 -Tjpg > exp/mono/bigram_graph/mono_bigram.jpg

# Decode unigram
steps/decode.sh --nj 4 exp/mono/unigram_graph data/dev exp/mono/decode_ug_dev
steps/decode.sh --nj 4 exp/mono/unigram_graph data/test exp/mono/decode_ug_test

# Decode bigram
steps/decode.sh --nj 4 exp/mono/bigram_graph data/dev exp/mono/decode_bg_dev
steps/decode.sh --nj 4 exp/mono/bigram_graph data/test exp/mono/decode_bg_test

# Alignment (ug and bg)
steps/align_si.sh --nj 4 data/train data/lang_test_unigram exp/mono exp/mono_ug_ali
steps/align_si.sh --nj 4 data/train data/lang_test_bigram exp/mono exp/mono_bg_ali

# Train triphone unigram
steps/train_deltas.sh 2000 10000 data/train data/lang_test_unigram exp/mono_ug_ali exp/tri_ug

utils/mkgraph.sh data/lang_test_unigram exp/tri_ug exp/tri_ug/graph
steps/decode.sh --nj 4 exp/tri_ug/graph data/test exp/tri_ug/decode

# Train triphone bigram
steps/train_deltas.sh 2000 10000 data/train data/lang_test_bigram exp/mono_bg_ali exp/tri_bg

utils/mkgraph.sh data/lang_test_bigram exp/tri_bg exp/tri_bg/graph
steps/decode.sh --nj 4 exp/tri_bg/graph data/test exp/tri_bg/decode
