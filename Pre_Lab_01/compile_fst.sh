#!/usr/bin/env bash

python3 pre_lab_01.py
# Create FSTs
fstcompile --isymbols=chars.syms --osymbols=chars.syms converter.txt > converter.fst
fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst
fstclosure acceptor.fst > acceptor_c.fst
fstrmepsilon acceptor_c.fst | fstdeterminize | fstminimize > acceptor_opt.fst
# fstcompile --isymbols=chars.syms --osymbols=chars.syms testing.txt > test.fst
# fstcompose acceptor_opt.fst converter.fst > comp.fst
# # fstarcsort comp.fst > comp_sort.fst
# fstcompose test.fst comp.fst > min_dist.fst
# # fstarcsort min_dist.fst > min_dist_s.fst
# fstshortestpath min_dist.fst > shortest.fst
