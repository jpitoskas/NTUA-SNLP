#!/usr/bin/env bash

python3 askisi_6.py

# Create Levenshtein Transducer
fstcompile --isymbols=chars.syms --osymbols=chars.syms transducer.txt > transducer.fst

fstclosure transducer.fst > transducer_closure.fst
# Draw Levenshtein Transducer
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait transducer.fst | dot -Tjpg > transducer.jpg
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait transducer_closure.fst | dot -Tjpg > transducer_closure.jpg
