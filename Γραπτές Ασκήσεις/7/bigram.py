from collections import defaultdict
import sys
import os
import re
# import subprocess
from math import *

def create_txt(output, words):
    inp = open(output, "w")
    for content in words:
        inp.write(content+"\n")
    inp.close()

L = ["lovely", "grand", "mother", "grandmother"]
book =  [3 * "lovely mother grand grandmother lovely grandmother grand mother " +
        "lovely mother grand grandmother lovely grandmother grand mother",
        8 * "lovely mother lovely grandmother lovely " +
        "lovely mother lovely grandmother lovely",
        2 * "mother grandmother mother " +
        "mother grandmother mother",
        1 * "lovely grand lovely"]

# print(book)

create_txt("lexicon.txt", L)
create_txt("sentences.txt", book)

# Create inputs of "lovelygrandmothergrand" based on L
input1 = "lovely grand mother grand".split(" ")
input2 = "lovely grandmother grand".split(" ")

print(input1)

def format_arc(src, dst, src_sym, dst_sym, w):
    return (str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')

def create_txt_fst(words, output):
    txt = open(output, 'w')
    s = 0
    for word in words:
        txt.write(
            format_arc(
                src=s, dst=s+1, src_sym=word, dst_sym=word, w=0))
        s += 1
    txt.write(str(s)+'\n')
    txt.close()

create_txt_fst(input1, 'input1.txt')
create_txt_fst(input2, 'input2.txt')
