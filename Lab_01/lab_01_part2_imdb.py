import sys
import os
import re
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# import subprocess
from math import *
import numpy as np
try:
    import glob2 as glob
except ImportError:
    import glob

data_dir = os.path.abspath('./aclImdb/')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM. (5000)
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 1000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

# Fix numpy random seed for reproducibility
SEED = 42
np.random.seed(42)

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

 def simple_lr_classify(X_tr, y_tr, X_test, y_test, description):
     # Helper function to train a logistic classifier and score on test data
     clf_LR = LogisticRegression().fit(X_tr, y_tr)
     score = clf_LR.score(X_test, y_test)
     print('Test score with', description, ': ', score)
     return clf_LR

# Load train sets
neg_train = read_samples(neg_train_dir, tokenize)
pos_train = read_samples(pos_train_dir, tokenize)

# Create corpus from train sets
corpus_train = create_corpus(pos_train, neg_train)

# print(corpus_train[0][3])
# print(corpus_train[1][3])

x_train = corpus_train[0]
y_train = corpus_train[1]

corpus_train_joined = list(itertools.chain.from_iterable(corpus_train[0]))

# print(corpus_train_joined)

# Transform using Count Vectorizer
BoW = CountVectorizer().fit_transform(corpus_train_joined)

# print(BoW)

clf_LR = LogisticRegression().fit(x_train, y_train)

# Transform using tf-idf Vectorizer
# BoW = TfidfVectorizer().fit_transform(corpus_train[0])

# print(pos_train)

# Load test set
# test = read_samples(test_dir, tokenize)

# print(test)
