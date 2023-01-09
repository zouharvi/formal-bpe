#!/usr/bin/env python3

import json
import orderedset
import itertools
import string
import argparse
from formal_bpe.model_exact_dyn import ExactDynBPE
from formal_bpe.model_exact_dfs import ExactDfsBPE
from formal_bpe.model_exact_brute_norm import ExactBruteNormBPE
from rich.progress import track

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=15)
args.add_argument("--alphabet-size", type=int, default=3)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=4)
args = args.parse_args()

alphabet = string.ascii_lowercase[:args.alphabet_size]

for example_length in range(5, args.example_length+1):
    iterator = map(
        "".join,
        itertools.product(alphabet, repeat=example_length)
    )
    paths_local = []

    # example = " ".join(open("data/CCrawl.de-en/dev.tok.en", "r").readlines()[:2])
    # print(example)
    for example_i, example in enumerate(map(''.join, track(iterator, total=1000))):
        if example_i == 1000:
            break
        # the letters must be sorted otherwise discard
        # this saves args.alphabet_size times the work
        example_letters = list(orderedset.OrderedSet(example))
        if not all(example_letters[i] <= example_letters[i + 1] for i in range(len(example_letters) - 1)):
            continue
        # the first n letters must be present to prevent duplicates like b,c,d,e..
        if not set(example_letters).issubset(alphabet[:len(set(example_letters))]):
            continue

        model = ExactBruteNormBPE(fix_overlap=True)
        result_exact, paths_val_brute = model.fit_greedy(
            example, T=args.merge_count,
        )

        model = ExactDfsBPE(fix_overlap=True)
        result_dyn = model.fit_greedy(
            example, T=args.merge_count,
        )

        # if example_i % 500 == 0 and example_i != 0:
        #     print(f"Dyn {sum(times_dyn):.1f}s")
        #     print(f"Brute {sum(times_brute):.1f}s")
        #     print(f"Ratio {sum(times_dyn)/sum(times_brute):.2f}")
        n_dyn = len(result_dyn)
        n_exact = len(result_exact)

        if n_dyn > n_exact:
            print("DYN", result_dyn)
            print("EXACT", result_exact)
            print("PROBLEM", n_dyn, n_exact)
            exit()

# with open(f"computed/merge_count/merge_{args.merge_count}.json", "w") as f:
#     json.dump(paths, f)