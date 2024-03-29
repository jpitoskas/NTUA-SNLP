import sys
import os
import re
import itertools
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from math import *
import numpy as np
from random import randint
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
MAX_NUM_SAMPLES = 25000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

# Fix numpy random seed for reproducibility
SEED = 42
np.random.seed(SEED)

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
    # Helper function to train a logistic regression classifier and score on test data
    LR = LogisticRegression(solver='liblinear')
    clf_LR = LR.fit(X_tr, y_tr)
    # LR.predict(X_test)
    print('Test score for LR classifier with', description, ':', clf_LR.score(X_test, y_test))
    return clf_LR

def simple_svm_classify(X_tr, y_tr, X_test, y_test, description):
    # Helper function to train a support vector machine classifier and score on test data
    SVM = SVC(gamma='auto')
    clf_SVM = SVM.fit(X_tr, y_tr)
    print('Test score for SVC classifier with', description, ':', clf_SVM.score(X_test, y_test))
    return clf_SVM

def simple_knn_classify(X_tr, y_tr, X_test, y_test, neigh, description):
    # Helper function to train a k-nearest neighbors classifier and score on test data
    KNN = KNeighborsClassifier(n_neighbors=neigh)
    clf_KNN = KNN.fit(X_tr, y_tr)
    print('Test score for KNN classifier (', neigh, 'neighbors ) with', description, ':', clf_KNN.score(X_test, y_test))
    return clf_KNN

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


# Count Vectorizer
cntVect = CountVectorizer()

# Transform using Count Vectorizer for train
BoW_cntVect_train = cntVect.fit_transform(corpus_train[0])
voc_train = list(cntVect.vocabulary_.keys())

# Transform using Count Vectorizer for test
BoW_cntVect_test = cntVect.transform(corpus_test[0])
voc_test = list(cntVect.vocabulary_.keys())

# Set train parameters
x_train = BoW_cntVect_train
y_train = corpus_train[1]

# Set test parameters
x_test = BoW_cntVect_test
y_test = corpus_test[1]

# Call train and evaluate function for Count Vectorizer
clf_cntVect = simple_lr_classify(x_train, y_train, x_test, y_test, "BoW with Count Vectorizer")


# Tf-iDf Vectorizer


TfiDfVect = TfidfVectorizer()

# Transform using tf-idf for train
BoW_TfiDf_train = TfiDfVect.fit_transform(corpus_train[0])

# Transform using tf-idf for test
BoW_TfiDf_test = TfiDfVect.transform(corpus_test[0])

# Set train parameters
x_train = BoW_TfiDf_train
# Set test parameters
x_test = BoW_TfiDf_test

# Call train and evaluate function for Tf-iDf Vectorizer
clf_TfiDf = simple_lr_classify(x_train, y_train, x_test, y_test, "BoW with Tf-iDf")


# Load pretrained w2v model from Pre Lab 01
# let vocabulary be the one in Step 9
model = Word2Vec.load('word2vec.model')
voc = model.wv.index2word
# get vector size
dim = model.vector_size

# Find percentage of "Out of Vocabulary" words
oov_cnt = 0
for word in voc_test:
    if word not in voc:
        oov_cnt += 1
oov = 100 * oov_cnt / len(voc_test)
print('Percentage of OOV words: ' + str(oov) + ' %')


df_train = {}
# Transform using Neural Bag of Words representation for train
# Also find weight for Tf-iDf
NBoW_train = np.zeros((len(corpus_train[0]), dim))
for i in range(len(corpus_train[0])):
    comment = corpus_train[0][i]
    words = comment.split(' ')
    comment_terms = len(comment)
    word_set = list(set(words))
    for term in word_set:
        if (term != '' and term in voc):
            if (term not in df_train.keys()):
                df_train[term] = 0
            df_train[term] += 1 / len(corpus_train[0])
    repr = np.zeros(dim)
    for word in words:
        if (word != ''):
            if (word in voc):
                repr = repr + model.wv[word]
    NBoW_train[i] = repr/len(words)


df_test = {}
# Transform using Neural Bag of Words representation for test
# Also find weight for Tf-iDf
NBoW_test = np.zeros((len(corpus_test[0]), dim))
for i in range(len(corpus_test[0])):
    comment = corpus_test[0][i]
    words = comment.split(' ')
    comment_terms = len(comment)
    word_set = list(set(words))
    for term in word_set:
        if (term != '' and term in voc):
            if (term not in df_test.keys()):
                df_test[term] = 0
            df_test[term] += 1 / len(corpus_test[0])
    repr = np.zeros(dim)
    for word in words:
        if (word != ''):
            if (word in voc):
                repr = repr + model.wv[word]
    NBoW_test[i] = repr/len(words)

# Call train and evaluate function for Neural Bag of Words representation
clf_cntVect = simple_lr_classify(NBoW_train, y_train, NBoW_test, y_test, "NBoW")

# Transform using Neural Bag of Words with Tf-iDf representation for train
NBoW_tfidf_train = np.zeros((len(corpus_train[0]), dim))
for i in range(len(corpus_train[0])):
    comment = corpus_train[0][i]
    comment_terms = len(comment)
    words = comment.split(' ')
    repr = np.zeros(dim)
    for word in words:
        if (word != ''):
            term_cnt = 0
            if (word in voc):
                for term in comment:
                    if (term == word):
                        term_cnt += 1
                tf_train = term_cnt / comment_terms
                repr = repr + model.wv[word] * (tf_train * log(1 / df_train[word]))
    NBoW_tfidf_train[i] = repr/len(words)

# Transform using Neural Bag of Words with Tf-iDf representation for train
NBoW_tfidf_test = np.zeros((len(corpus_test[0]), dim))
for i in range(len(corpus_test[0])):
    comment = corpus_test[0][i]
    comment_terms = len(comment)
    words = comment.split(' ')
    repr = np.zeros(dim)
    for word in words:
        if (word != ''):
            term_cnt = 0
            if (word in voc):
                for term in comment:
                    if (term == word):
                        term_cnt += 1
                tf_test = term_cnt / comment_terms
                repr = repr + model.wv[word] * (tf_test * log(1 / df_test[word]))
    NBoW_tfidf_test[i] = repr/len(words)

# Call train and evaluate function for Neural Bag of Words with Tf-iDf representation
clf_cntVect = simple_lr_classify(NBoW_tfidf_train, y_train, NBoW_tfidf_test, y_test, "NBoW with Tf-iDf")


# # Load pretrained Google model over all GoogleNews
# google_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)
# google_voc = google_model.wv.index2word
#
# # For 10 random words in vocabulary
# for i in range(10):
#     rnd = randint(0, len(voc))
#     while (not(voc[rnd] in google_voc)):
#         rnd = randint(0, len(voc))
#     # get most similar word
#     sim = model.wv.most_similar(voc[rnd])
#     sim_google = google_model.wv.most_similar(voc[rnd])
#     print('For word: '+voc[rnd])
#     print('Most similar (in text): '+sim[0][0])
#     print('Most similar (Google): '+sim_google[0][0])
#     print()
#
# # Bonus Step 19
#
# simple_svm_classify(BoW_TfiDf_train, y_train, BoW_TfiDf_test, y_test, 'BoW with Tf-iDf')
# simple_knn_classify(BoW_TfiDf_train, y_train, BoW_TfiDf_test, y_test, 5, 'BoW with Tf-iDf')
