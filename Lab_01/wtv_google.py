import sys
import re
import os
import gensim
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from random import randint

NUM_W2V_TO_LOAD = 1000000

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
        return embedding_matrix, model.wv.index2word, word2idx

model = Word2Vec.load('word2vec.model')

# get ordered vocabulary list
voc = model.wv.index2word

google_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)
google_voc = google_model.wv.index2word
# print(google_voc)
# get vector size
dim = model.vector_size

# For 10 random words in vocabulary
for i in range(10):
    rnd = randint(0, len(voc))
    while (not(voc[rnd] in google_voc)):
        rnd = randint(0, len(voc))
    # get most similar word
    sim = model.wv.most_similar(voc[rnd])
    sim_google = google_model.wv.most_similar(voc[rnd])
    print('For word: '+voc[rnd])
    print('Most similar (in text): '+sim[0][0])
    print('Most similar (Google): '+sim_google[0][0])
    print()

# print(model.wv['india'] == model['india'])
corpus = [  voc[1] + ' ' + voc[5] + ' ' + voc[80] + ' ' + voc[178],
            voc[17] + ' ' + voc[42],
            voc[324] + ' ' + voc[284] + ' ' + voc[1500] + ' ' + voc[312]  ]

embedding_matrix, voc, word2idx = to_embeddings_Matrix(model)

for i in
