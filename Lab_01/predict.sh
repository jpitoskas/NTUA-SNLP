#!/usr/bin/env bash


# python3 test.py |
#     fstcompile --isymbols=chars.syms --osymbols=chars.syms |
#     fstcompose - ${1} |
#     fstshortestpath |
#     fstrmepsilon |
#     fsttopsort |
# fstprint -osymbols=chars.syms |
cut -f4 ${1} |
grep -v "<epsilon>" |
head -n -1 |
tr -d '\n'
