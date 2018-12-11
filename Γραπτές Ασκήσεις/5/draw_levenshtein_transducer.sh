#!/usr/bin/env bash

python3 ex5_levenshtein_transducer.py

# Create Levenshtein Transducer
fstcompile --isymbols=chars.syms --osymbols=chars.syms levenshtein_transducer.txt > levenshtein_transducer.fst

# Draw Levenshtein Transducer
fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait levenshtein_transducer.fst | dot -Tjpg > levenshtein_transducer.jpg

# Create FSA
fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst
fstrmepsilon acceptor.fst | fstdeterminize | fstminimize > acceptor_opt.fst
fstarcsort acceptor_opt.fst > acceptor_opt_sorted.fst

# Create testing FSTs
  fstcompile --isymbols=chars.syms --osymbols=chars.syms input.txt > input.fst

  # Compose acceptor and converter and sort afterwards to create orthograph
  fstcompose input.fst levenshtein_transducer.fst > input_conv.fst
  fstarcsort input_conv.fst > input_conv_sorted.fst

  # Draw composed transducer
  # fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait input_conv_sorted.fst | dot -Tjpg > input_conv.jpg

  # Compose test with orthograph and sort
  fstcompose input_conv_sorted.fst acceptor_opt_sorted.fst > ortho_input.fst
  # fstarcsort ortho_input.fst > ortho_input_sorted.fst
  fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait ortho_input_sorted.fst | dot -Tjpg > ortho_input.jpg
  # Draw composed transducer with orthograph and test
  # fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait ortho_input_sorted.fst | dot -Tjpg > ortho_input.jpg
