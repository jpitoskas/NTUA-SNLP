#!/bin/bash
source ./kaldi-master/egs/wsj/s5/path.sh
cd ./kaldi-master/egs/wsj/s5

# MFCCs
steps/make_mfcc.sh
steps/compute_cmvn_stats.sh
