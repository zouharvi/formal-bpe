#!/usr/bin/env python3

import orderedset
import itertools
import string
import argparse
from rich.progress import track
from itertools import takewhile
import multiprocess as mp

args = argparse.ArgumentParser()
args.add_argument("--example-length", type=int, default=3)
args.add_argument("--alphabet-size", type=int, default=3)
args.add_argument("--beam-size", type=int, default=5)
args.add_argument("--merge-count", type=int, default=3)
args = args.parse_args()

# we can cache this to be faster
# this is terribly slow
def get_signature(merge):
    if type(merge) is str:
        return (1, merge)
    else:
        count1, sig1 = get_signature(merge[0])
        count2, sig2 = get_signature(merge[1])
        return (count1+count2, sig1+sig2)

def common_prefix(a, b):
    return [i[0] for i in takewhile(lambda x: x[0] == x[1], zip(a, b))]

def canonize_order(merges, example):
    # remove duplicates
    merges = list(set(merges))
    merges.sort(key=lambda x: x[0])
    return merges

def canonize_order_worst(merges, example):
    # remove duplicates
    merges = list(set(merges))
    best_merge = max(itertools.permutations(merges), key=lambda x: len(encode(example, x)))
    return best_merge

def merges_only(merges):
    return [x[1] for x in merges]

class Lettuce:
    def __init__(self, merges):
        # assume that the merge operations are ordered well
        self.merges = list(merges)

    def join(self, other, example):
        new_merges = self.merges + other.merges
        new_merges = canonize_order_worst(new_merges, example)
        return Lettuce(new_merges)

    def meet(self, other):
        new_merges = common_prefix(self.merges, other.merges)
        return Lettuce(new_merges)

def merge_dependents(merge):
    if type(merge) is tuple and type(merge[0]) is str and type(merge[1]) is str:
        return {merge}
    elif type(merge) is str:
        return set()
    else:
        return merge_dependents(merge[0])|merge_dependents(merge[1])

def is_valid(merges):
    for merge in merges:
        dependents = merge_dependents(merge)
        if any(d not in merges for d in dependents):
            return False
    return True

def apply_merge_slow(token, pair):
    ys_word = []
    i = 0
    N = len(token)
    while i < N:
        if i < N - 1 and (token[i], token[i + 1]) == pair:
            ys_word.append(pair)
            i += 2
        else:
            ys_word.append(token[i])
            i += 1
    return ys_word

def generate_merges(string):
    merges = set(string)
    for i in range(args.merge_count):
        # deduplicate
        merges = set(merges)
        merges |= set(itertools.product(set(merges), set(merges)))

    return [(get_signature(x), x) for x in merges if len(x) >= 2]

def encode(token, merges):
    for merge in merges:
        # don't use the signature
        token = apply_merge_slow(token, merge[1])
    return token


base_merges = generate_merges("aaab")

alphabet = string.ascii_lowercase[:args.alphabet_size]
iterator = map(
    "".join,
    itertools.product(alphabet, repeat=args.example_length)
)
total = args.alphabet_size**args.example_length

# for example in map(
#     ''.join,
#     iterator
#     # track(iterator, total=total)
# ):
for example in ["aabaaaaaba"]:
    print(example)
    # the letters must be sorted otherwise discard
    # this saves args.alphabet_size times the work
    example_letters = list(orderedset.OrderedSet(example))
    if not all(example_letters[i] <= example_letters[i + 1] for i in range(len(example_letters) - 1)):
        continue
    # the first n letters must be present to prevent duplicates like b,c,d,e..
    if not set(example_letters).issubset(alphabet):
        continue

    lattuces = [Lettuce([x]) for x in base_merges if is_valid([x[1]])]
    for i in range(args.merge_count):
        print(i, len(lattuces))
        lattices_new = []

        def check_local(x):
            lattice1_i, lattice1 = x
            print(lattice1_i)
            f1 = encode(example, lattice1.merges)
            for lattice2 in lattuces[lattice1_i+1:]:
                f2 = encode(example, lattice2.merges)
                lattice_meet = lattice1.meet(lattice2)
                lattice_join = lattice1.join(lattice2, example)

                f_meet = encode(example, lattice_meet.merges)
                f_join = encode(example, lattice_join.merges)

                # check submodularity
                if not (len(f1)+len(f2) <= len(f_meet) + len(f_join)):
                    print("s", example)
                    print("M1", merges_only(lattice1.merges))
                    print("M2", merges_only(lattice2.merges))
                    print("M1.meet(M2)", merges_only(lattice_meet.merges))
                    print("M1.join(M2)", merges_only(lattice_join.merges))
                    print("len M1", len(f1))
                    print("len M2", len(f2))
                    print("len M1.meet(M2)", len(f_meet))
                    print("len M1.join(M2)", len(f_join))
                    exit()
        

        with mp.Pool(20) as pool:
            pool.map(check_local, enumerate(track(lattuces)))

        for lattice1 in lattuces:
            for merge in base_merges:
                new_merges = lattice1.merges + [merge]
                if is_valid([x[1] for x in new_merges]):
                    lattices_new.append(Lettuce(canonize_order_worst(new_merges, example)))

        print(": merges", len(base_merges))
        print(":", len(lattuces), "+", len(lattices_new))
        lattuces += lattices_new