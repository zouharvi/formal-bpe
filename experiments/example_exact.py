#!/usr/bin/env python3

import orderedset
import itertools
import string
import argparse
from formal_bpe.model_exact_dyn import ExactDynBPE
from formal_bpe.model_exact_brute import ExactBruteBPE
from formal_bpe.model_exact_brute_norm import ExactBruteNormBPE
from formal_bpe.model_slow import SlowBPE
from rich.progress import track

args = argparse.ArgumentParser()
args.add_argument("--example-length-start", type=int, default=None)
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=2)
args = args.parse_args()

if args.example_length_start is None:
    args.example_length_start = args.example_length

alphabet = string.ascii_lowercase[:args.alphabet_size]

for length in range(args.example_length_start, args.example_length + 1):
    iterator = map(
        "".join,
        itertools.product(alphabet, repeat=length)
    )
    total = args.alphabet_size**length
    print(f"\n{length}\n")
    for example in map(
        ''.join,
        track(iterator, total=total)
    ):

        # the letters must be sorted otherwise discard
        # this saves args.alphabet_size times the work
        example_letters = list(orderedset.OrderedSet(example))
        if not all(example_letters[i] <= example_letters[i + 1] for i in range(len(example_letters) - 1)):
            continue
        # the first n letters must be present to prevent duplicates like b,c,d,e..
        if not set(example_letters).issubset(alphabet[:len(set(example_letters))]):
            continue

        # model = ExactDynBPE(fix_overlap=True)
        # result_dyn = model.fit_greedy(
        #     example, T=args.merge_count,
        # )

        model = ExactBruteNormBPE(fix_overlap=True)
        result_exact = model.fit_greedy(
            example, T=args.merge_count,
        )

        # n_beam = len(result_dyn)
        # n_exact = len(result_exact)

        # if n_beam != n_exact:
        #     print(n_beam, n_exact)