import sys
import os
import errno
import re
from collections import defaultdict

# Create directories
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
usc_dir = os.path.abspath('./kaldi-master/egs/usc/')
local_dir = os.path.join(usc_dir, 'local/')
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
local_score_kaldi_dir = os.path.join(local_dir, 'score_kaldi.sh')
score_kaldi_dir = os.path.abspath('.kaldi-master/egs/wsj/s5/steps/score_kaldi.sh')
try:
    os.symlink(score_kaldi_dir, local_score_kaldi_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        os.remove(local_score_kaldi_dir)
        os.symlink(score_kaldi_dir, local_score_kaldi_dir)
    else:
        raise e
usc_steps_dir = os.path.join(usc_dir, 'steps')
steps_dir = os.path.abspath('.kaldi-master/egs/wsj/s5/steps')
try:
    os.symlink(steps_dir, usc_steps_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        os.remove(usc_steps_dir)
        os.symlink(steps_dir, usc_steps_dir)
    else:
        raise e
usc_utils_dir = os.path.join(usc_dir, 'utils')
utils_dir = os.path.abspath('.kaldi-master/egs/wsj/s5/utils')
try:
    os.symlink(utils_dir, usc_utils_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        os.remove(usc_utils_dir)
        os.symlink(utils_dir, usc_utils_dir)
    else:
        raise e
conf_dir = os.path.join(usc_dir, 'conf/')
if not os.path.exists(conf_dir):
    os.makedirs(conf_dir)
lang_dir = os.path.join(usc_dir, 'data/lang/')
if not os.path.exists(lang_dir):
    os.makedirs(lang_dir)
dict_dir = os.path.join(usc_dir, 'data/local/dict/')
if not os.path.exists(dict_dir):
    os.makedirs(dict_dir)
lm_tmp_dir = os.path.join(usc_dir, 'data/local/lm_tmp/')
if not os.path.exists(lm_tmp_dir):
    os.makedirs(lm_tmp_dir)
nist_lm_dir = os.path.join(usc_dir, 'data/local/nist_lm/')
if not os.path.exists(nist_lm_dir):
    os.makedirs(nist_lm_dir)


# MPAAA

def create_dict(path, words):
    non_sil_dir = os.path.join(path, "nonsilence_phones.txt")
    lex_dir = os.path.join(path, "lexicon.txt")
    sil_dir = os.path.join(path, "silence_phones.txt")
    opt_sil_dir = os.path.join(path, "optional_silence.txt")
    non_sil = open(non_sil_dir, "w")
    lex = open(lex_dir, "w")
    sil = open(sil_dir, "w")
    opt_sil = open(opt_sil_dir, "w")
    for content in words:
        lex.write(content+" "+content+"\n")
        if (content == "sil"):
            sil.write(content+"\n")
            opt_sil.write(content+"\n")
        else:
            non_sil.write(content+"\n")
    non_sil.close()
    lex.close()
    sil.close()
    opt_sil.close()

def tokenize(s):
    s = s.strip()
    s = s.lower()
    # Keep lower/upper case characters, numbers
    regex = re.compile("[^a-z']")
    s = regex.sub(' ', s)
    s = s.replace('\n','')
    s = re.sub(' +',' ', s)
    s = s.split(' ')
    return s

def load_phonemes():
    phonemes = {}
    phonemes = defaultdict(lambda:"", phonemes)
    all_phonemes = []
    with open(lexicon_dir, 'r') as f:
        line = f.readline()
        while line:
            line = line.replace('\n','')
            line = line.replace("\t", " ")
            line = line.split("  ")
            if (line[0] == "<oov> <oov>"):
                line[0] = "<oov>"
                line.append("<oov>")
            all_phonemes += line[1].split(" ")
            phonemes[line[0].lower()] = line[1]
            line = f.readline()
    unique_phonemes = list(set(all_phonemes))
    unique_phonemes = sorted(unique_phonemes)
    print(len(phonemes))
    return phonemes, unique_phonemes


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
            sentence = tokenize(sentences[int(id) - 1])
            phone_sent = ""
            for word in sentence:
                if (word != ""):
                    phone_sent += phonemes[word] + " "
            text.write(utt_id+" "+"sil "+phone_sent+"sil\n")

            line = f.readline()
            cnt += 1
    trans.close()
    uttids.close()
    utt2spk.close()
    wavscp.close()
    text.close()

# main

phonemes, unique_phonemes = load_phonemes()
create_files(uttrain_dir, train_dir)
create_files(uttest_dir, test_dir)
create_files(utvalid_dir, dev_dir)
create_dict(dict_dir, unique_phonemes)
