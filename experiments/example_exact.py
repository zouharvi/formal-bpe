#!/usr/bin/env python3

import orderedset
import itertools
import string
import argparse
from formal_bpe.model_exact_dyn import ExactDynBPE
from formal_bpe.model_exact_dfs import ExactDFSBPE
from formal_bpe.model_exact_dfs_mem import ExactDFSMemBPE
from formal_bpe.model_exact_brute import ExactBruteBPE
from formal_bpe.model_exact_brute_norm import ExactBruteNormBPE
from formal_bpe.model_slow import SlowBPE
import time

args = argparse.ArgumentParser()
args.add_argument("--merge-count", type=int, default=3)
args = args.parse_args()

times_dyn = []
times_brute = []

example = " ".join(open("data/CCrawl.de-en/dev.tok.en", "r").readlines()[:2])
print(example)

print("DFS+Memoization")
time_start = time.time()
model = ExactDFSMemBPE(fix_overlap=True)
result_dfsmem = model.fit_greedy(
    example, T=args.merge_count,
)
times_dyn.append(time.time() - time_start)
    

print("DFS")
time_start = time.time()
model = ExactDFSBPE(fix_overlap=True)
result_dfs = model.fit_greedy(
    example, T=args.merge_count,
)
times_brute.append(time.time() - time_start)


n_dfsmem = len(result_dfsmem)
n_brute = len(result_dfs)
print(result_dfsmem)
print(result_dfs)

print("lengths", n_dfsmem, n_brute)

print(f"DFS {sum(times_dyn):.1f}s")
print(f"DFS+Mem {sum(times_brute):.1f}s")
print(f"Ratio {sum(times_dyn)/sum(times_brute):.2f}")
