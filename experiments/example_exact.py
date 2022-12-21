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
import time

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=8)
args.add_argument("--alphabet-size", type=int, default=2)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=2)
args = args.parse_args()

alphabet = string.ascii_lowercase[:args.alphabet_size]

times_dyn = []
times_brute = []

iterator = map(
    "".join,
    itertools.product(alphabet, repeat=args.example_length)
)
total = args.alphabet_size**args.example_length
print(f"\n{args.example_length}\n")

example = " ".join(open("data/CCrawl.de-en/dev.tok.en", "r").readlines()[:2])
print(example)
# for example_i, example in enumerate(map(''.join, track(iterator, total=total / 2))):
    # the letters must be sorted otherwise discard
    # this saves args.alphabet_size times the work
    # example_letters = list(orderedset.OrderedSet(example))
    # if not all(example_letters[i] <= example_letters[i + 1] for i in range(len(example_letters) - 1)):
    #     continue
    # # the first n letters must be present to prevent duplicates like b,c,d,e..
    # if not set(example_letters).issubset(alphabet[:len(set(example_letters))]):
    #     continue

time_start = time.time()
model = ExactBruteNormBPE(fix_overlap=True)
result_exact = model.fit_greedy(
    example, T=args.merge_count,
)
times_brute.append(time.time() - time_start)

time_start = time.time()
model = ExactDynBPE(fix_overlap=True)
result_dyn, _ = model.fit_greedy(
    example, T=args.merge_count,
)
times_dyn.append(time.time() - time_start)

# if example_i % 500 == 0 and example_i != 0:
#     print(f"Dyn {sum(times_dyn):.1f}s")
#     print(f"Brute {sum(times_brute):.1f}s")
#     print(f"Ratio {sum(times_dyn)/sum(times_brute):.2f}")
    

n_dyn = len(result_dyn)
n_exact = len(result_exact)

if n_dyn > n_exact:
    print(n_dyn, n_exact)

print(f"Dyn {sum(times_dyn):.1f}s")
print(f"Brute {sum(times_brute):.1f}s")
print(f"Ratio {sum(times_dyn)/sum(times_brute):.2f}")
