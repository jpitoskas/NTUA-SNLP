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

def syms(alphabet, output):
    # alphabet.sort()
    out = open(output, 'w')
    out.write('<epsilon> 0\n')
    for index in range(len(alphabet)):
        out.write(str(alphabet[index])+' '+str(index+1)+'\n')
    out.close()

def testing(word, output):
    testing = open(output, 'w')
    letters = list(word)
    s = 0
    for i in range(len(letters)):
        testing.write(
            format_arc(
                src=s, dst=s+1, src_sym=letters[i], dst_sym=letters[i], w=0))
        s += 1
    testing.write(str(s)+'\n')
    testing.close()


def levenshtein_transducer(alphabet, w, output):
    transducer_fd = open(output, 'w')
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
            transducer_fd.write(
                format_arc(
                    0, 0, src_sym, dst_sym, weight))
    transducer_fd.write(str(0)+'\n')
    transducer_fd.close()

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

alphabet = ['A', 'G', 'C', 'T', 'E', 'F']
str1 = 'AECAGEF'
str2 = 'TETCGAG'
accepted_word = [str1]

syms(alphabet, 'chars.syms')
testing(str2, 'input.txt')
levenshtein_transducer(alphabet, 1, 'levenshtein_transducer.txt')
acceptoras(accepted_word, 0, 'acceptor.txt')
