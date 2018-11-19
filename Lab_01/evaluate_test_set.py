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

def read_file(path):
    output = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            output += line
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
print(correct)
input = read_file(sys.argv[2])
print(input)
output = read_file(sys.argv[3])
print(output)

correct_cnt = 0
for i in range(len(input)):
    if (correct[input[i]] == output):
        correct_cnt += 1

per = correct_cnt / len(input)

print("Percentage of correct predicted words:", per, "%")
