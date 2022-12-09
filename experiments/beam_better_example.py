#!/usr/bin/env python3

raise Exception("This part has not yet been updated to the object-oriented design")

from collections import Counter
import itertools
import string
import argparse
import orderedset
from arsenal.iterextras import take
from faster_bpe.model import FasterBPE
from faster_bpe.model_slow import SlowBPE
from faster_bpe.utils import pairs_in_list, pretty_seq

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args = args.parse_args()

def find_max_and_merge(xs):
    if len(xs) <= 1: return xs
    c = Counter(pairs_in_list(xs))
    pair, _ = c.most_common()[0]
    return SlowBPE.apply_merge_slow(xs, pair)

def greedy(xs, T=None):
    old = xs
    t = 0
    while True:
        t += 1
        if T is not None and t > T: break
        new = find_max_and_merge(old)
        if old == new: break
        old = new
    return old

def possible_splits_gen(xs):
    if len(xs) <= 1:
        yield xs
        return

    c = Counter(pairs_in_list(xs))
    for pair in sorted(c, key=c.__getitem__, reverse=True):
        yield SlowBPE.apply_merge_slow(xs, pair)

def beam_search(xs, T, B):
    beam = [xs]
    for t in range(T):
        bs = beam + [ys for xs in beam for ys in take(B, possible_splits_gen(xs))]
        beam = sorted(bs, key=len)[:B]
    return min(beam, key=len)


for example in map(
    ''.join,
    itertools.product(string.ascii_lowercase[:args.alphabet_size], repeat=args.example_length)
):
    # the letters must be sorted otherwise discard
    # this saves args.alphabet_size times the work
    example_letters = list(orderedset.OrderedSet(example))
    if not all(example_letters[i] <= example_letters[i+1] for i in range(len(example_letters) - 1)):
        continue

    # discard examples which don't use all letters
    if len(example_letters) != args.alphabet_size:
        continue

    # print(example, example_letters)

    bpe_greedy = FasterBPE(example)
    result_greedy = greedy(example, 2)

    bpe_beam = FasterBPE(example)
    result_beam = beam_search(example, 2, B=2)

    result_greedy_pretty = pretty_seq(result_greedy)
    result_beam_pretty = pretty_seq(result_beam)
    if "(aa)" in result_greedy_pretty:
        continue

    if len(result_greedy) > len(result_beam):
        print("Example:    ", example)
        print(f"Greedy:      ({len(result_greedy)})", result_greedy_pretty)
        print(f"Beam search: ({len(result_beam)})", result_beam_pretty)
        print("====")
    # print(len(result_greedy), len(result_beam))