#!/bin/bash
source ./kaldi-master/egs/usc/path.sh
cd ./kaldi-master/egs/usc/

# MFCCs
for ste in train dev test; do
  utils/fix_data_dir.sh data/$ste
  steps/make_mfcc.sh data/$ste
  # steps/compute_cmvn_stats.sh
done
