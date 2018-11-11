#!/usr/bin/env bash

python3 pre_lab_01.py
# Create FSTs
fstcompile --isymbols=chars.syms --osymbols=chars.syms converter.txt > converter.fst
fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst
fstrmepsilon acceptor.fst | fstdeterminize | fstminimize > acceptor_opt.fst
# fstcompile --isymbols=chars.syms --osymbols=chars.syms testing.txt > test.fst
# fstcompose test.fst converter.fst > comp.fst
# fstarcsort comp.fst > comp_sort.fst
# fstcompose acceptor_opt.fst comp_sort.fst > min_dist.fst
# fstarcsort min_dist.fst > min_dist_s.fst
# fstshortestpath min_dist_s.fst > shortest.fst
