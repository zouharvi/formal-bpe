#!/usr/bin/env python3

from collections import Counter
import itertools
import string
import argparse
import orderedset
from arsenal.iterextras import take
from faster_bpe.model import FasterBPE
from faster_bpe.model_slow import SlowBPE
from faster_bpe.utils import pretty_seq

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args = args.parse_args()

for example in map(
    ''.join,
    itertools.product(string.ascii_lowercase[:args.alphabet_size], repeat=args.example_length)
):
    bpe_slow = SlowBPE(fix_overlap=False)
    result_slow = bpe_slow.fit_greedy(example, 2)

    bpe_faster = FasterBPE()
    result_faster = bpe_faster.fit_greedy(example, 2)

    assert len(result_slow) <= len(result_faster)

    if len(result_slow) != len(result_faster):
        print(f"{example}")
        print(len(result_slow), "|||", len(result_faster))
        print(result_slow, "|||", result_faster)
