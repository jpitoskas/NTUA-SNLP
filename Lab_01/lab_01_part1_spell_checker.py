from collections import defaultdict
import sys
import os
import re
# import subprocess
from math import *

def format_arc(src, dst, src_sym, dst_sym, w):
    # out = open('test.fst', 'w')
    # out.write(str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')
    # out.close()
    return (str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')

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

def syms(alphabet, output):
    alphabet.sort()
    out = open(output, 'w')
    out.write('<epsilon> 0\n')
    for index in range(len(alphabet)):
        out.write(str(alphabet[index])+' '+str(index+1)+'\n')
    out.close()

def converter(alphabet, w, output):
    converter = open(output, 'w')
    for i in range(len(alphabet)+1):
        for j in range(len(alphabet)+1):
            if (i == 0):
                src_sym = '<epsilon>'
            else:
                src_sym=alphabet[i-1]
            if (j ==0):
                dst_sym = '<epsilon>'
            else:
                dst_sym=alphabet[j-1]
            if (i == j):
                weight = 0
            else:
                weight = w
            converter.write(
                format_arc(
                    0, 0, src_sym, dst_sym, weight))
    converter.write(str(0)+'\n')
    converter.close()

def acceptoras(tokens, weight, output):
    acceptor = open(output, 'w')
    s = 1
    acceptor.write(
        format_arc(
            src=0, dst=0, src_sym="<epsilon>", dst_sym="<epsilon>", w=weight))
    for token in tokens:
        letters = list(token)
        for i in range(0, len(letters)):
            if (i == 0):
                acceptor.write(
                    format_arc(
                        src=0, dst=s, src_sym=letters[i], dst_sym=letters[i], w=weight))
            else:
                acceptor.write(
                    format_arc(
                        src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=weight))
                s += 1
            # if (i == len(letters) - 1):
            #     acceptor.write(
            #         format_arc(
            #             src=s, dst=0, src_sym='<epsilon>', dst_sym='<epsilon>', w=weight))
        if (len(letters) != 0):
            acceptor.write(str(s)+'\n')
            s += 1
    acceptor.close()

def acceptor_word_level(tokens, dictionary, output):
    acceptor = open(output, 'w')
    s = 1
    acceptor.write(
        format_arc(
            src=0, dst=0, src_sym="<epsilon>", dst_sym="<epsilon>", w=0))
    for token in tokens:
        letters = list(token)
        for i in range(0, len(letters)):
            if (i == 0):
                acceptor.write(
                    format_arc(
                        src=0, dst=s, src_sym=letters[i], dst_sym=letters[i], w=-log(dictionary[token], 2)))
            else:
                acceptor.write(
                    format_arc(
                        src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=0))
                s += 1
            # if (i == len(letters) - 1):
            #     acceptor.write(
            #         format_arc(
            #             src=s, dst=0, src_sym='<epsilon>', dst_sym='<epsilon>', w=weight))
        if (len(letters) != 0):
            acceptor.write(str(s)+'\n')
            s += 1
    acceptor.close()

def acceptor_unigram(tokens, dictionary, output):
    acceptor = open(output, 'w')
    s = 1
    acceptor.write(
        format_arc(
            src=0, dst=0, src_sym="<epsilon>", dst_sym="<epsilon>", w=0))
    for token in tokens:
        letters = list(token)
        for i in range(0, len(letters)):
            if (i == 0):
                acceptor.write(
                    format_arc(
                        src=0, dst=s, src_sym=letters[i], dst_sym=letters[i], w=-log(dictionary[letters[i]], 2)))
            else:
                acceptor.write(
                    format_arc(
                        src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=-log(dictionary[letters[i]], 2)))
                s += 1
            # if (i == len(letters) - 1):
            #     acceptor.write(
            #         format_arc(
            #             src=s, dst=0, src_sym='<epsilon>', dst_sym='<epsilon>', w=weight))
        if (len(letters) != 0):
            acceptor.write(str(s)+'\n')
            s += 1
    acceptor.close()

# def runsubprocesses():
#     p1 = subprocess.Popen("fstcompile --isymbols=chars.syms --osymbols=chars.syms converter.txt > converter.fst")
#     # p1.wait()
#     p2 = subprocess.Popen("fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst")
#     # p2.wait()
#     p3 = subprocess.Popen("fstrmepsilon acceptor.fst | fstdeterminize | fstminimize > acceptor_opt.fst")
#     # p3.wait()

# Find absolute path
path = os.path.abspath(sys.argv[1])

res = read_file(path, tokenize)

# print(len(res))

# Unique words of a list
tokens = list(set(res))

# print(len(tokens))

all_words = ''.join(tokens)

# Unique letters of a list
alphabet = list(set(all_words))

# Create the chars.syms file
syms(alphabet, 'chars.syms')

converter(alphabet, 1, 'converter.txt')

acceptoras(tokens, 0, 'acceptor.txt')

# dictionary to match each word of the book to its likelihood of occurrence
word_probability_dict = {}
word_probability_dict = defaultdict(lambda:0, word_probability_dict)
num_of_letters = 0
for i in range(len(res)):
    num_of_letters = num_of_letters + len(res[i])
    word_probability_dict[res[i]] = word_probability_dict[res[i]] + 1/len(res)

# dictionary to match each character of the book to its likelihood of occurrence
char_probability_dict = {}
char_probability_dict = defaultdict(lambda:0, char_probability_dict)
for i in range(len(res)):
    for j in range(len(res[i])):
        char_probability_dict[res[i][j]] = char_probability_dict[res[i][j]] + 1/num_of_letters

# find a mean word weight
word_weights = 0
for token in tokens:
    word_weights = word_weights - log(word_probability_dict[token], 2)
mean_word_weight = word_weights/len(tokens)

# Create transducer for mean weight of words
converter(alphabet, mean_word_weight, "converter_words.txt")

# find a mean char weight
char_weights = 0
for char in alphabet:
    char_weights = char_weights - log(char_probability_dict[char], 2)
mean_char_weight = char_weights/len(alphabet)

# Create transducer for mean weight of letters
converter(alphabet, mean_char_weight, "converter_unigram.txt")

acceptor_word_level(tokens, word_probability_dict, "acceptor_word_level.txt")
acceptor_unigram(tokens, char_probability_dict, "acceptor_unigram.txt")

# dictionary to match each pair of adjacent chars of the book to its likelihood of occurrence
pairs = []
for i in range(len(res)):
    for j in range(len(res[i]) - 1):
        char_pair = str(res[i][j]) + str(res[i][j+1])
        pairs.append(char_pair)

# Unique pairs of chars of a list
unique_pairs = list(set(pairs))

pair_probability_dict = {}
pair_probability_dict = defaultdict(lambda:0, pair_probability_dict)
for i in range(len(pairs)):
    pair_probability_dict[pairs[i]] = pair_probability_dict[pairs[i]] + 1/len(pairs)
