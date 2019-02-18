#!/usr/bin/env bash

# Create lexicon and sentences
python3 bigram.py

# Create symbols for n-gram
ngramsymbols <sentences.txt >sentences.syms

# Create FAR binary
farcompilestrings -symbols=sentences.syms -keep_symbols=1 sentences.txt >sentences.far

# Create n-gram count
ngramcount -order=2 sentences.far >sentences.cnts

# Make n-gram
ngrammake --method=katz --backoff=true sentences.cnts >sentences.mod

# Draw FST
fstdraw --isymbols=sentences.syms --osymbols=sentences.syms -portrait sentences.mod | dot -Gdpi=300 -Tjpg > bigram_all.jpg

# Create FST for input1
fstcompile --isymbols=sentences.syms --osymbols=sentences.syms input1.txt > input1.fst

# Create FST for input2
fstcompile --isymbols=sentences.syms --osymbols=sentences.syms input2.txt > input2.fst

# Compose acceptor and converter and sort afterwards to create orthograph
fstcompose input1.fst sentences.mod | fstarcsort > input1_conv_sorted.fst

# Draw FST1
fstdraw --isymbols=sentences.syms --osymbols=sentences.syms -portrait input1_conv_sorted.fst | dot -Gdpi=300 -Tjpg > input1_comp.jpg

# Find minimum distance (Max probability -> Shortest path in (neg log) weighted graph)
fstshortestpath input1_conv_sorted.fst > min_dist_input1.fst

# Draw min dist FST1
fstdraw --isymbols=sentences.syms --osymbols=sentences.syms -portrait min_dist_input1.fst | dot -Gdpi=300 -Tjpg > input1_min_dist.jpg

# Compose acceptor and converter and sort afterwards to create orthograph
fstcompose input2.fst sentences.mod | fstarcsort > input2_conv_sorted.fst

# Find minimum distance (Max probability -> Shortest path in (neg log) weighted graph)
fstshortestpath input2_conv_sorted.fst > min_dist_input2.fst

# Draw min dist FST2
fstdraw --isymbols=sentences.syms --osymbols=sentences.syms -portrait min_dist_input2.fst | dot -Gdpi=300 -Tjpg > input2_min_dist.jpg

# Draw FST2
fstdraw --isymbols=sentences.syms --osymbols=sentences.syms -portrait input2_conv_sorted.fst | dot -Gdpi=300 -Tjpg > input2_comp.jpg


# Random sentence
# ngramrandgen --max_sents=1 sentences.mod | farprintstrings

ngramprint --ARPA sentences.mod > sentences.ARPA
head -40 sentences.ARPA
