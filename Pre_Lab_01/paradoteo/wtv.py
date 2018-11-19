import sys
import re
import os
import gensim
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

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

# Find absolute path
path = os.path.abspath("Around the World in 80 Days, by Jules Verne.txt")

res = read_file(path, tokenize)

# Unique words of a list
tokens = list(set(res))

w2v = get_tmpfile("word2vec.model")
# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(tokens, window=5, size=100, workers=4)
model.save("word2vec.model")
model.train(tokens, total_examples=len(tokens), epochs=1000)

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# get most similar words
sim = model.wv.most_similar('holmes')

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx
