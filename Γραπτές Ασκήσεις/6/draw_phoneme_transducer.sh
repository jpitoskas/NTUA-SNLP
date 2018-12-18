#!/usr/bin/env bash

python3 askisi_6.py

# Create Levenshtein Transducer
fstcompile --isymbols=chars.syms --osymbols=chars.syms transducer.txt > transducer.fst

fstclosure transducer.fst > transducer_closure.fst
fstrmepsilon transducer_closure.fst | fstdeterminize | fstminimize > transducer_test.fst
# fstarcsort acceptor_opt.fst > acceptor_opt_sorted.fst
# Draw Levenshtein Transducer
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait transducer.fst | dot -Gdpi=1000 -Tjpg > transducer.jpg
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait transducer_closure.fst | dot -Gdpi=1000 -Tjpg > transducer_closure.jpg
