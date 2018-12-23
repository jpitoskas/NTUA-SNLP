#!/bin/bash
source ./kaldi-master/egs/wsj/s5/path.sh
cd ./kaldi-master/egs/usc/

# Train
steps/train_mono.sh
steps/train_deltas.sh
# steps/train_lda_mllt.sh

# Generate graph
utils/mkgraph.sh

# Alignment
steps/align_si.sh

# Decode
steps/decode.sh
