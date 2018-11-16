import sys
import os
import re
import itertools
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
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
MAX_NUM_SAMPLES = 2
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
    # s = s.split(' ')
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
    clf_LR = LogisticRegression(solver='liblinear').fit(X_tr, y_tr)
    # clf_LR.predict(X_test)
    print('Test score with', description, ': ', clf_LR.score(X_test, y_test))
    # return clf_LR

# Load pretrained w2v model from Pre Lab 01
model = Word2Vec.load('word2vec.model')
voc = model.wv.index2word

# Load train sets
neg_train = read_samples(neg_train_dir, tokenize)
pos_train = read_samples(pos_train_dir, tokenize)

# Load test sets
neg_test = read_samples(neg_test_dir, tokenize)
pos_test = read_samples(pos_test_dir, tokenize)

# Create corpus from train sets
corpus_train = create_corpus(pos_train, neg_train)

# Create corpus from test sets
corpus_test = create_corpus(pos_test, neg_test)

# corpus_train_joined = list(itertools.chain.from_iterable(corpus_train[0]))

# print(x_train[1])

# Transform using Count Vectorizer for train
BoW_train = CountVectorizer().fit_transform(corpus_train[0])

# Transform using Count Vectorizer for test
BoW_test = CountVectorizer().fit_transform(corpus_test[0])

x_train = BoW_train
y_train = corpus_train[1]

# print(x_train.shape)
# print(len(y_train))

x_test = BoW_test
y_test = corpus_test[1]

# print(x_test.shape)
# print(len(y_test))

simple_lr_classify(x_train, y_train, x_test, y_test, "bourda")

# Transform using tf-idf Vectorizer
# BoW = TfidfVectorizer().fit_transform(corpus_train[0])

# print(pos_train)

# Load test set
# test = read_samples(test_dir, tokenize)

# print(test)
