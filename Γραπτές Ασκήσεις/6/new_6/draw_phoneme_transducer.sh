#!/usr/bin/env bash

python3 ex6.py

fstcompile --isymbols=chars.syms --osymbols=chars.syms chars.stxt > chars.fst
# Create Levenshtein Transducer
fstcompile --isymbols=chars.syms --osymbols=chars.syms newtransducer.stxt > transducer.fst

fstclosure chars.fst > chars_closure.fst
# fstrmepsilon chars_closure.fst | fstdeterminize > chars_min.fst

# fstclosure transducer.fst > transducer_closure.fst
# fstrmepsilon transducer_closure.fst | fstdeterminize | fstminimize > transducer_test.fst

# fstarcsort acceptor_opt.fst > acceptor_opt_sorted.fst
# Draw Levenshtein Transducer
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait chars_closure.fst | dot -Gdpi=500 -Tjpg > chars.jpg
# fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait transducer_closure.fst | dot -Gdpi=500 -Tjpg > transducer_closure.jpg
