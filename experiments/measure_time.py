#!/usr/bin/env python3

import json
import time
import argparse
from formal_bpe.model import FasterBPE
from formal_bpe.model_slow import SlowBPE
from external_model_wrap import SentencePiece

args = argparse.ArgumentParser()
args.add_argument(
    "-ns", nargs="+", type=int,
    default=[1_000, 10_000, 100_000, 1_000_000]
)
args.add_argument("-m", "--model", default="slow")
args.add_argument("-l", "--logfile", default="computed/time_slow.jsonl")
args.add_argument("-v", "--vocab-size", type=int, default=8000)
args = args.parse_args()

if args.model == "slow_notok":
    def model_factory(): return SlowBPE(fix_overlap=False, tokenize=False)
elif args.model == "faster_notok":
    def model_factory(): return FasterBPE(tokenize=False)
elif args.model == "slow_tok":
    def model_factory(): return SlowBPE(fix_overlap=False, tokenize=True)
elif args.model == "faster_tok":
    def model_factory(): return FasterBPE(tokenize=True)
elif args.model == "sentencepiece":
    def model_factory(): return SentencePiece()

fout = open(args.logfile, "w")

def word_stats(data):
    words = data.split()
    return len(words), len(set(words))

data = list(open("data/CCrawl.de-en/train.tok.all").readlines())
for n in args.ns:
    data_local = "".join(data[:n]).rstrip("\n")
    word_count, word_uniq_count = word_stats(data_local)
    model = model_factory()

    print(f"n={n}")
    start_time = time.time()
    result = model.fit_greedy(data_local, args.vocab_size, progress_bar=True)
    total_time = time.time() - start_time

    # todo: also store observed vocabulary
    logline = {
        "model": args.model,
        "vocab_size": 8000,
        "n_line": len(data[:n]),
        "n_word": word_count,
        "n_word_uniq": word_uniq_count,
        "time": total_time,
        "enc_uniq": len({w for line in result for w in line}),  
        "enc_len": sum(len(line) for line in result)
    }
    print(logline)
    fout.write(json.dumps(logline) + "\n")
    fout.flush()