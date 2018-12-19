import sys
import os
import re
from collections import defaultdict

data_dir = os.path.abspath('./slp_lab2_data/')
filesets_dir = os.path.join(data_dir, 'filesets/')
uttest_dir = os.path.join(filesets_dir, 'test_utterances.txt')
uttrain_dir = os.path.join(filesets_dir, 'train_utterances.txt')
utvalid_dir = os.path.join(filesets_dir, 'validation_utterances.txt')
wavs_dir = os.path.join(data_dir, 'wav/')
f1_dir = os.path.join(wavs_dir, 'f1')
f5_dir = os.path.join(wavs_dir, 'f5')
m1_dir = os.path.join(wavs_dir, 'm1')
m3_dir = os.path.join(wavs_dir, 'm3')
transcription_dir = os.path.join(data_dir, 'transcription.txt')
lexicon_dir = os.path.join(data_dir, 'lexicon.txt')
test_dir = os.path.abspath('./kaldi-master/egs/usc/data/test/')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
train_dir = os.path.abspath('./kaldi-master/egs/usc/data/train/')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
dev_dir = os.path.abspath('./kaldi-master/egs/usc/data/dev/')
if not os.path.exists(dev_dir):
    os.makedirs(dev_dir)

# MPAAA

def load_phonemes():
    phonemes = {}
    phonemes = defaultdict(lambda:"", phonemes)
    with open(lexicon_dir, 'r') as f:
        line = f.readline()
        while line:
            line = line.replace('\n','')
            line = line.split("\t ")
            if (line[0] == "<oov>  <oov>"):
                line[0] = "<oov>"
                line.append("<oov>")
            # print(line[1])
            phonemes[line[0].lower()] = line[1]
            line = f.readline()
    print(len(phonemes))


def create_files(src, dest):
    trans = open(transcription_dir, 'r')
    sentences = trans.readlines()

    uttids_dir = os.path.join(dest, 'uttids.txt')
    uttids = open(uttids_dir, 'w')
    utt2spk_dir = os.path.join(dest, 'utt2spk.txt')
    utt2spk = open(utt2spk_dir, 'w')
    wavscp_dir = os.path.join(dest, 'wav.scp')
    wavscp = open(wavscp_dir, 'w')
    text_dir = os.path.join(dest, 'text.txt')
    text = open(text_dir, 'w')

    with open(src, 'r') as f:
        line = f.readline()
        cnt = 1
        while line:
            line = line.replace('\n','')
            line = line.split('_')
            speaker = line[2]
            id = line[3]
            utt_id = "utterance_id_"+str(cnt)
            uttids.write(utt_id+"\n")
            utt2spk.write(utt_id+" "+str(speaker)+"\n")
            wavdir = os.path.join(wavs_dir, speaker)
            wavdir += "/usctimit_ema_"+str(speaker)+"_"+str(id)+".wav"
            wavscp.write(utt_id+" "+wavdir+"\n")
            text.write(utt_id+" "+sentences[int(id) - 1])

            line = f.readline()
            cnt += 1

    trans.close()
    uttids.close()
    utt2spk.close()
    wavscp.close()
    text.close()

phonemes = load_phonemes()
# create_files(uttrain_dir, train_dir)
# create_files(uttest_dir, test_dir)
# create_files(utvalid_dir, dev_dir)
