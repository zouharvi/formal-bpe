#!/usr/bin/env python3

from collections import Counter
import itertools
import string
import argparse
import orderedset
from arsenal.iterextras import window, take
from faster_bpe.model import FasterBPE, SlowBPE, merge

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=7)
args.add_argument("--alphabet-size", type=int, default=2)
args = args.parse_args()

def _reference(xs):
    if len(xs) <= 1: return xs
    c = Counter(window(xs, 2))
    pair, _ = c.most_common()[0]
    return merge(xs, pair)

def greedy(xs, T=None):
    old = xs
    t = 0
    while True:
        t += 1
        if T is not None and t > T: break
        new = _reference(old)
        if old == new: break
        old = new
    return old

def possible_splits(xs):
    if len(xs) <= 1:
        yield xs
        return    
    c = Counter(window(xs, 2))
    for pair in sorted(c, key=c.__getitem__, reverse=True):
#        if c[pair] <= 1: continue   # drop pairs with count <= 1
        yield merge(xs, pair)

def beam_search(xs, T, B):
    beam = [xs]
    for t in range(T):
        bs = beam + [ys for xs in beam for ys in take(B, possible_splits(xs))]
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

    if len(result_greedy) > len(result_beam):
        print("Example:    ", example)
        print(f"Greedy:      ({len(result_greedy)})", result_greedy)
        print(f"Beam search: ({len(result_beam)})", result_beam)
        print("====")
    # print(len(result_greedy), len(result_beam))