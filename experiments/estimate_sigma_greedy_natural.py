#!/usr/bin/env python3

import tqdm
import argparse
import itertools
import string
from formal_bpe.model_exact_dfs import ExactDFSBPE
from formal_bpe.model_slow import SlowBPE

args = argparse.ArgumentParser()
args.add_argument("--merge-count", type=int, default=2)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--example-length", type=int, default=6)
args = args.parse_args()

sigma = 0
EXAMPLES = [
    "the expeditious russet-hued vulpine creature elegantly leaps over the languid, lackadaisical canine companion",
    "the quick brown fox jumped over the lazy dog",
    "If Fantasy Hockey actually lived up to its name, every team would have Henrik Lundqvist and Joffrey Lupul on it I have a moral code, but I haven't figured out how to read it yet We say we are walking the dog, but the dog always leads A tagline for an airline: Take the High Road INjuries always keep you OUT of things. Visticula If you wake up with a giant zit, you are really facing your fears when you look in the mirror",
    'Are there Out-of-Stock photos? Gafuffle North America should be called Russia since people are always moving so fast. Gralitica You should "listen to my mixtape" (check out the rest of my portfolio)',
]
for example in EXAMPLES:
    model_opt = ExactDFSBPE(fix_overlap=True)
    model_gen = SlowBPE(fix_overlap=True)

    example_opt, merges_opt = model_opt.fit_greedy(
        example, T=args.merge_count,
    )
    util_opt = len(example) - len(example_opt)

    example_new, merges_new = model_gen.fit_greedy(
        example, T=args.merge_count, output_merge_seq=True
    )
    example_new = example_new[0]
    
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