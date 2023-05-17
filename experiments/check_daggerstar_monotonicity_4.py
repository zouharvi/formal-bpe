#!/usr/bin/env python3

import itertools
import string
import argparse
from formal_bpe.model_exact_dfs_mem import ExactDFSMemBPE
from formal_bpe.model_exact_dyn import ExactDynBPE
from formal_bpe.model_slow import SlowBPE
import tqdm

from formal_bpe.utils import debug_flat_seq

args = argparse.ArgumentParser()
args.add_argument("--merge-count", type=int, default=2)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--example-length", type=int, default=7)
args = args.parse_args()

alphabet = string.ascii_lowercase[:args.alphabet_size]

iterator = map(
    "".join,
    itertools.product(alphabet, repeat=args.example_length)
)
total = args.alphabet_size**(args.example_length-1)
for example in map(
    ''.join,
    tqdm.tqdm(iterator, total=total)
):
    model = ExactDFSMemBPE(fix_overlap=True)
    result_opt, merges_opt = model.fit_greedy(
        example, T=args.merge_count,
        output_merge_seq=True
    )
    merges_opt = [x[1] for x in merges_opt]
        
    model = SlowBPE(fix_overlap=True)
    result_greedy, merges_greedy = model.fit_greedy(
        example, T=args.merge_count, debug_output=True,
        output_merge_seq=True
    )
    # idk
    result_greedy = result_greedy[0]
    best_last_count = 0
    for merge in merges_opt:
        last_v = debug_flat_seq([merge])
        greedy_last_count = sum(1 for x in result_greedy if last_v == x)
        best_last_count = max(greedy_last_count, best_last_count)

    n_opt = len(example)-len(result_opt)
    n_greedy = len(example)-len(result_greedy)

    avg_diff = 1/args.merge_count*(n_opt-n_greedy)
    if best_last_count < avg_diff:
        print(example)
        print(n_opt, result_opt, merges_opt, sep=" ||| ")
        print(n_greedy, result_greedy, merges_greedy, sep=" ||| ")
        print()
        
    continue
    if n_opt > n_greedy:
        model_blank = ExactDynBPE(fix_overlap=True)
        tokens_concat = example
        for pair in merges_greedy + merges_opt:
            tokens_concat = model_blank.apply_merge_slow(tokens_concat, pair)
        n_concat = len(example)-len(tokens_concat)
        if n_concat < n_opt:
            print(example)
            print(n_opt, result_opt, merges_opt, sep=" ||| ")
            print(n_greedy, result_greedy, merges_greedy, sep=" ||| ")
            print(n_concat, tokens_concat, sep=" ||| ")
            print()


    # print("lengths", n_dfsmem, n_brute)