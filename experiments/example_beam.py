#!/usr/bin/env python3

import orderedset
import itertools
import string
import argparse
from arsenal.iterextras import take
from formal_bpe.model_slow_beam import SlowBeamBPE
from formal_bpe.model_slow_insert import SlowInsertBPE
from formal_bpe.model_exact_brute import ExactBruteBPE
from formal_bpe.model_exact_greedy import ExactGreedyBPE
from formal_bpe.model_exact_dfs_mem import ExactDFSMemBPE
from formal_bpe.model_slow import SlowBPE
from rich.progress import track

args = argparse.ArgumentParser()
args.add_argument("--example-length-start", type=int, default=None)
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=4)
args = args.parse_args()

if args.example_length_start is None:
    args.example_length_start = args.example_length

min_ratio = 1
alphabet = string.ascii_lowercase[:args.alphabet_size]

for length in range(args.example_length_start, args.example_length + 1):
    iterator = map(
        "".join,
        itertools.product(alphabet, repeat=length)
    )
    total = args.alphabet_size**(length-1)
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

        model = ExactGreedyBPE(fix_overlap=True)
        result_greedy, indecision = model.fit_greedy(
            example, T=args.merge_count,
            # debug_output=True, indecision_output=True
        )
        # result_greedy = result_greedy[0]

        model = ExactDFSMemBPE(fix_overlap=True)
        result_exact = model.fit_greedy(
            example, T=args.merge_count,
        )

        # model = SlowBeamBPE()
        # result_beam = model.fit_beam_search(
        #     example, T=args.merge_count,
        #     B=args.beam_size
        # )

        n_beam = len(example) - len(result_exact)
        n_greedy = len(example) - len(result_greedy)
        # print(f"Ratio: {n_greedy/ n_beam:.2f}")


        if min_ratio >= n_greedy / n_beam and n_beam / n_greedy != 1:
            min_ratio = n_greedy/n_beam
        # if len(result_exact) < len(result_greedy):
            print("Example:  ", example, "indecision", indecision)
            print(f"Greedy:   ({len(result_greedy)})", result_greedy)
            print(f"Other:    ({len(result_exact)})", result_exact)
            print(f"Ratio:     {n_greedy/ n_beam:.2f}")
            print("====")

        # assert len(result_exact) <= len(result_greedy)

        # if len(result_exact) < len(result_greedy) and not indecision:
        #     print("Example:    ", example, "indecision", indecision)
        #     print(f"Greedy:      ({len(result_greedy)})", result_greedy)
        #     print(f"Beam search: ({len(result_beam)})", result_beam)
        #     print("====")
