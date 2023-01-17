#!/usr/bin/env python3

import orderedset
import itertools
import string
import argparse
from formal_bpe.model_exact_dyn import ExactDynBPE
from formal_bpe.model_exact_brute import ExactBruteBPE
from formal_bpe.model_exact_greedy import ExactGreedyBPE
from formal_bpe.model_exact_perm import SlowExactPermBPE
from formal_bpe.model_exact_brute_norm import ExactBruteNormBPE
from formal_bpe.model_slow import SlowBPE
from rich.progress import track
import time

args = argparse.ArgumentParser()
args.add_argument("--example-length-start", type=int, default=None)
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--merge-count", type=int, default=2)
args = args.parse_args()

if args.example_length_start is None:
    args.example_length_start = args.example_length

alphabet = string.ascii_lowercase[:args.alphabet_size]

times_1 = []
times_2 = []

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

        start_time = time.time()
        model = ExactDFSMemBPE(fix_overlap=True)
        result_exact = model.fit_greedy(
            example, T=args.merge_count,
        )
        times_1.append(time.time()-start_time)

        start_time = time.time()
        model = ExactGreedyBPE(fix_overlap=True)
        result_greedy, type_1_indecision = model.fit_greedy(
            example, T=args.merge_count,
        )
        times_2.append(time.time()-start_time)

        n_beam = len(result_exact)
        n_exact = len(result_greedy)

        if n_beam != n_exact and not type_1_indecision:
            print(n_beam, n_exact)
            print(example)
            print("Exact", result_exact)
            print("Greedy", result_greedy)
            exit()

print(f"Time exactbrute: {sum(times_1):.1f}s")
print(f"Time exactgreedy: {sum(times_2):.1f}s")