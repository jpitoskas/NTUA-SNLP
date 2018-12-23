#!/bin/bash
source ./kaldi-master/egs/wsj/s5/path.sh
cd ./kaldi-master/egs/usc/

# MFCCs
for ste in train dev test; do
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf data/$ste
  # steps/compute_cmvn_stats.sh
done
