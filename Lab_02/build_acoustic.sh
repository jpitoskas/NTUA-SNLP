#!/bin/bash
source ./kaldi-master/egs/usc/path.sh
cd ./kaldi-master/egs/usc/

# MFCCs
for ste in train dev test; do
  utils/fix_data_dir.sh data/$ste || exit 1;
  steps/make_mfcc.sh --nj 4 data/$ste || exit 1;
  steps/compute_cmvn_stats.sh data/$ste || exit 1;
done
