import sys
import os
import re

def format_arc(src, dst, src_sym, dst_sym, w):
    # out = open('test.fst', 'w')
    # out.write(str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')
    # out.close()
    return (str(src)+' '+str(dst)+' '+str(src_sym)+' '+str(dst_sym)+' '+str(w)+'\n')

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

testing(sys.argv[1], 'test_input.txt')
