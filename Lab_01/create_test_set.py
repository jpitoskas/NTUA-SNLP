import sys
import os
import re
from random import randint

def identity_preprocess(string):
    return string

def read_test(path, preprocess = None):
    if preprocess is None:
        preprocess = identity_preprocess
    processed = []
    with open(path, 'r') as f:
        line = f.readline()
        # print(line)
        while line:
            ps = preprocess(line)
            processed += ps[1:]
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

def create_test_file(words, output):
    testing = open(output, 'w')
    for word in words:
        testing.write(word+'\n')
    testing.close()

# Pick 20 random words for evaluation
def random_test_file(words, output):
    testing = open(output, 'w')
    for _ in range(20):
        rnd = randint(0, len(words))
        testing.write(words[rnd]+'\n')
    testing.close()

test = read_test(sys.argv[1], tokenize)
# print(test)

create_test_file(test, 'test_set.txt')
random_test_file(test, 'random_test_set_20.txt')
