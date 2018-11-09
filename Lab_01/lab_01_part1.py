from collections import defaultdict
import sys
import os
import re
import subprocess
from math import *

# Dynamic Programming algorithm to find Levenshtein Distance between two words
def LevenshteinDistance(str1, str2, m, n):
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]

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

def syms(alphabet):
    alphabet.sort()
    out = open('chars.syms', 'w')
    out.write('<epsilon> 0\n')
    for index in range(len(alphabet)):
        out.write(str(alphabet[index])+' '+str(index+1)+'\n')
    out.close()

def converter(alphabet, w):
    converter = open('converter.txt', 'w')
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



def acceptoras(tokens):
    acceptor = open('acceptor.txt', 'w')
    for token in tokens:
        s = 1
        letters = list(token)
        for i in range(0, len(letters)):
            acceptor.write(
                format_arc(
                    src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=0))
            s += 1
            if (i == len(letters) - 1):
                acceptor.write(
                    format_arc(
                        src=s, dst=0, src_sym='<epsilon>', dst_sym='<epsilon>', w=0))
        s += 1
        acceptor.write(str(s)+'\n')
    acceptor.close()


# def runsubprocesses():
#     p1 = subprocess.Popen("fstcompile --isymbols=chars.syms --osymbols=chars.syms converter.txt > converter.fst")
#     # p1.wait()
#     p2 = subprocess.Popen("fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst")
#     # p2.wait()
#     p3 = subprocess.Popen("fstrmepsilon acceptor.fst | fstdeterminize | fstminimize > acceptor_opt.fst")
#     # p3.wait()

# Find absolute path
path = os.path.abspath("Around the World in 80 Days, by Jules Verne.txt")

res = read_file(path, tokenize)

# test = read_file(sys.argv[1], tokenize)


# Unique words of a list
tokens = list(set(res))


all_words = ''.join(tokens)

# Unique letters of a list
alphabet = list(set(all_words))

# Create the chars.syms file
syms(alphabet)

converter(alphabet, 1)

acceptoras(tokens)

# runsubprocesses()

# testing(test)

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


# find a mean char weight
char_weights = 0
for char in alphabet:
    char_weights = char_weights - log(char_probability_dict[char], 2)
mean_char_weight = char_weights/len(alphabet)

# converter(alphabet, mean_char_weight)
