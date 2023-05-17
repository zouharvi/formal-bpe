#!/usr/bin/env python3

import tqdm
import argparse
import itertools
import string
from formal_bpe.model_exact_dfs import ExactDFSBPE
from formal_bpe.generator_dfs_merges import GeneratorDFSMerges

args = argparse.ArgumentParser()
args.add_argument("--merge-count", type=int, default=2)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--example-length", type=int, default=6)
args = args.parse_args()

alphabet = string.ascii_lowercase[:args.alphabet_size]
iterator = map(
    "".join,
    itertools.product(alphabet, repeat=args.example_length)
)
total = args.alphabet_size**(args.example_length - 1)
sigma = 0
for example in map(
    ''.join,
    tqdm.tqdm(iterator, total=total)
):
    model_opt = ExactDFSBPE(fix_overlap=True)
    model_gen = GeneratorDFSMerges(fix_overlap=True)

    example_opt, merges_opt = model_opt.fit_greedy(
        example, T=args.merge_count,
    )
    util_opt = len(example) - len(example_opt)

    for (merges_new, example_new) in model_gen.yield_all_mergeseq(example, T=args.merge_count):
        x = example
        for merge in merges_new + merges_opt:
            x = ExactDFSBPE.apply_merge_slow(x, merge)
        util_new = len(example) - len(example_new)
        util_join = len(example) - len(x)
        sigma_local = 1-(util_join-util_opt)/util_new
        if sigma_local > sigma:
            print("optimal", merges_opt, example_opt)
            print("new", merges_new, example_new)
            print("joined", merges_new, example_new)
            print(
                example,
                "optimal util:", util_opt,
                "new util:", util_new,
                "sigma local", sigma_local,
            )
            print()
            sigma = max(sigma, sigma_local)