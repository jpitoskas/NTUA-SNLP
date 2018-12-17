from collections import defaultdict
import sys
import os
import re
# import subprocess
from math import *


L = ["lovely", "grand", "mother", "grandmother"]
book =  [3 * "lovely mother grand grandmother lovely grandmother grand mother " +
        "lovely mother grand grandmother lovely grandmother grand mother",
        8 * "lovely mother lovely grandmother lovely " +
        "lovely mother lovely grandmother lovely",
        2 * "mother grandmother mother " +
        "mother grandmother mother",
        1 * "lovely grand lovely"]

lex = open("lexicon.txt", "w")
for word in L:
    lex.write(word+"\n")
lex.close()

sent = open("sentences.txt", "w")
for sentence in book:
    sent.write(sentence+"\n")
sent.close()

# print(book)

# # dictionary to match each pair of adjacent chars of the book to its likelihood of occurrence
# pairs = []
# for i in range(len(book)):
#     for j in range(len(book[i]) - 1):
#         char_pair = str(book[i][j]) + str(book[i][j+1])
#         pairs.append(char_pair)
#
# # Unique pairs of chars of a list
# unique_pairs = list(set(pairs))
#
# pair_probability_dict = {}
# pair_probability_dict = defaultdict(lambda:0, pair_probability_dict)
# for i in range(len(pairs)):
#     pair_probability_dict[pairs[i]] = pair_probability_dict[pairs[i]] + 1/len(pairs)
