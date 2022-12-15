#!/usr/bin/env python3

import itertools
import string
import argparse
from formal_bpe.model import FasterBPE
from formal_bpe.model_slow import SlowBPE

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args = args.parse_args()

for example in map(
    ''.join,
    itertools.product(string.ascii_lowercase[:args.alphabet_size], repeat=args.example_length)
):
    bpe_slow = SlowBPE(fix_overlap=False)
    result_slow = bpe_slow.fit_greedy(example, 2, debug_output=True)[0]

    bpe_faster = FasterBPE()
    result_faster = bpe_faster.fit_greedy(example, 2)[0]

    assert len(result_slow) <= len(result_faster)

    # the reason why faster bpe sometimes gives worse results is because it does not implement the overlap fix
    # and due to the max heap, always attempts to first merge earlier pairs (i.e. pairs)
    # for two steps of merges, that will always be suboptimal (but actually is better from application perspective)
    if len(result_slow) != len(result_faster):
        print(f"\n{example}")
        print(len(result_slow), "|||", len(result_faster))
        print(result_slow, "|||", result_faster)

example="The lazy brown fox jumped over the quick frog.\nThe brown jumper with the froggo suits me well."

print("\nWithout tokenization")
model = SlowBPE(fix_overlap=False, tokenize=False)
result = model.fit_greedy(example, 10)
print(sum(len(line) for line in result), result)

print("\nWith tokenization")
model = SlowBPE(fix_overlap=False, tokenize=True)
result = model.fit_greedy(example, 10)
print(sum(len(line) for line in result), result)

print("\nFaster without tokenization")
model = FasterBPE(tokenize=False)
result = model.fit_greedy(example, 10)
print(sum(len(line) for line in result), result)

print("\nFaster with tokenization")
model = FasterBPE(tokenize=True)
result = model.fit_greedy(example, 10)
print(sum(len(line) for line in result), result)