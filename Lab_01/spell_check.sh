#!/usr/bin/env bash

# Create a wrong-spelled evaluation test set
python3 create_test_set.py "evaluation_set8.txt"
# Create spell checker FST
python3 lab_01_part1_spell_checker.py "Around the World in 80 Days, by Jules Verne.txt"

# Create FSTs
fstcompile --isymbols=chars.syms --osymbols=chars.syms converter.txt > converter.fst
fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor.txt > acceptor.fst

# Draw acceptor before optimizations
# fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait acceptor.fst | dot -Tjpg > acceptor.jpg

# fstclosure acceptor.fst > acceptor_c.fst
fstrmepsilon acceptor.fst | fstdeterminize | fstminimize > acceptor_opt.fst
fstarcsort acceptor_opt.fst > acceptor_opt_sorted.fst
# Draw optimal acceptor
# fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait acceptor_opt.fst | dot -Tjpg > acceptor_opt.jpg
> prediction_20rnd.txt

while read word; do
  # Use spell checker to predict every word in the test set
  python3 predict_test_set.py "$word"

  # Create testing FSTs
  fstcompile --isymbols=chars.syms --osymbols=chars.syms test_input.txt > input.fst

  # Compose acceptor and converter and sort afterwards to create orthograph
  fstcompose input.fst converter.fst > input_conv.fst
  fstarcsort input_conv.fst > input_conv_sorted.fst

  # Draw composed transducer
  # fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait input_conv_sorted.fst | dot -Tjpg > input_conv.jpg

  # Compose test with orthograph and sort
  fstcompose input_conv_sorted.fst acceptor_opt_sorted.fst > ortho_input.fst
  fstarcsort ortho_input.fst > ortho_input_sorted.fst
  # fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait ortho_input_sorted.fst | dot -Tjpg > ortho_input_sorted.jpg
  # Draw composed transducer with orthograph and test
  # fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait ortho_input_sorted.fst | dot -Tjpg > ortho_input.jpg

  # Find minimum distance
  fstshortestpath ortho_input_sorted.fst > min_dist.fst

  # Draw/Print minimum distance transducer
  fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait min_dist.fst | dot -Tjpg > min_dist.jpg
  fsttopsort min_dist.fst | fstprint --isymbols=chars.syms --osymbols=chars.syms - > min_dist.txt

  cut -f4 min_dist.txt |
  grep -v "<epsilon>" |
  head -n -1 |
  tr -d '\n' >> prediction_20rnd.txt
  printf '\n' >> prediction_20rnd.txt

done < "random_test_set_20.txt"

python3 evaluate_test_set.py "evaluation_set8.txt" "random_test_set_20.txt" "prediction_20rnd.txt"
