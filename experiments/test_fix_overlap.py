#!/usr/bin/env python3

import itertools
import string
import argparse
from faster_bpe.model_slow import SlowBPE
from faster_bpe.utils import debug_flat_seq

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=9)
args.add_argument("--alphabet-size", type=int, default=2)
args = args.parse_args()

for example in map(
    ''.join,
    itertools.product(string.ascii_lowercase[:args.alphabet_size], repeat=args.example_length)
):
    print(f"{example}")
    bpe_slow = SlowBPE()
    result_slow = bpe_slow.fit_greedy(example, 2)

    bpe_slow_fixed = SlowBPE(fix_overlap=True)
    result_slow_fixed = bpe_slow_fixed.fit_greedy(example, 2)

# 0.5M chars
corpus = open("data/CCrawl.de-en/train.tok.all").read(500_000)

bpe_slow = SlowBPE()
result_slow = bpe_slow.fit_greedy(corpus, 32)
bpe_slow_fixed = SlowBPE(fix_overlap=True)
result_slow_fixed = bpe_slow_fixed.fit_greedy(corpus, 32)

# does not happen in data
print(len(result_slow), len(result_slow_fixed))