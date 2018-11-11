import sys
import os
import re
# import subprocess
from math import *
import numpy as np
try:
    import glob2 as glob
except ImportError:
    import glob

data_dir = './aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

# Fix numpy random seed for reproducibility
SEED = 42
np.random.seed(42)

def identity_preprocess(string):
    return string

def read_file(path, preprocess = None):
    if preprocess is None:
        preprocess = identity_preprocess
    processed = []
    with open(path, 'r') as f:
        line = f.readline()
        # print(line)
        while line:
            ps = preprocess(line)
            processed += ps
            line = f.readline()
        return processed

def tokenize(s):
    s = s.strip()
    s = s.lower()
    # Keep lower/upper case characters, numbers and spaces
    regex = re.compile('[^a-z ]')
    s = regex.sub(' ', s)
    s = s.replace('\n',' ')
    s = re.sub(' +',' ', s)
    s = s.split(' ')
    return s

def testing(words):
    for word in words:
        testing = open('testing.txt', 'w')
        letters = list(word)
        s = 0
        for i in range(len(letters)):
            testing.write(str(s)+' '+str(s+1)+' '+str(letters[i])+' '+str(letters[i])+'\n')
            s += 1
        testing.write(str(s)+'\n')
    testing.close()

def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, 'r') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data

def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return list(corpus[indices]), list(y[indices])

# Find absolute path
path = os.path.abspath("Around the World in 80 Days, by Jules Verne.txt")

res = read_file(path, tokenize)

# test = read_file(sys.argv[1], tokenize)
