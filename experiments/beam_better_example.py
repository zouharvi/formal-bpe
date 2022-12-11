#!/usr/bin/env python3

from collections import Counter
import itertools
import string
import argparse
import orderedset
from arsenal.iterextras import take
from faster_bpe.model_slow_beam import SlowBeamBPE
from faster_bpe.model_slow import SlowBPE
from faster_bpe.utils import pairs_in_list, flat_seq
from rich.progress import track

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=12)
args.add_argument("--alphabet-size", type=int, default=4)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=3)
args = args.parse_args()

iterator = map("".join, itertools.product(string.ascii_lowercase[:args.alphabet_size], repeat=args.example_length))
total = args.alphabet_size**args.example_length

for example in map(
    ''.join,
    track(iterator, total=total)
):
    bpe_greedy = SlowBPE(fix_overlap=True)
    result_greedy, indecision = bpe_greedy.fit_greedy(
        example, T=args.merge_count,
        debug_output=True, indecision_output=True
    )
    result_greedy = result_greedy[0]


    bpe_beam = SlowBeamBPE()
    result_beam = bpe_beam.fit_beam_search(
        example, T=args.merge_count,
        B=args.beam_size
    )

    if len(result_greedy) > len(result_beam):
        print("Example:    ", example, "indecision", indecision)
        print(f"Greedy:      ({len(result_greedy)})", result_greedy)
        print(f"Beam search: ({len(result_beam)})", result_beam)
        print("====")

# print(len(result_greedy), len(result_beam))


# # the letters must be sorted otherwise discard
# # this saves args.alphabet_size times the work
# example_letters = list(orderedset.OrderedSet(example))
# if not all(example_letters[i] <= example_letters[i+1] for i in range(len(example_letters) - 1)):
#     continue

# # discard examples which don't use all letters
# if len(example_letters) != args.alphabet_size:
#     continue
# # print(example, example_letters)