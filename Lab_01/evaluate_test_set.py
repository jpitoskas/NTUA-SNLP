import sys
import os
import re

def identity_preprocess(string):
    return string

correct = {}
def create_correct_dict(path, preprocess = None):
    if preprocess is None:
        preprocess = identity_preprocess
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            ps = preprocess(line)
            for i in range(1, len(ps)):
                correct[ps[i]] = ps[0]
            line = f.readline()

def read_file(path, preprocess = None):
    if preprocess is None:
        preprocess = identity_preprocess
    output = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            ps = preprocess(line)
            output += ps
            line = f.readline()
        return output

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

create_correct_dict(sys.argv[1], tokenize)
input = read_file(sys.argv[2], tokenize)
# print(input)
output = read_file(sys.argv[3], tokenize)
# print(output)

correct_cnt = 0
for i in range(len(input)):
    if (correct[input[i]] == output[i]):
        correct_cnt += 1

per = 100 * correct_cnt / len(input)

print("Percentage of correct predicted words:", per, "%")
