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
ngrammake sentences.cnts >sentences.mod

# Random sentence
ngramrandgen --max_sents=1 sentences.mod | farprintstrings

ngramprint --ARPA sentences.mod >sentences.ARPA
head -40 sentences.ARPA
