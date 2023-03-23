#!/usr/bin/env python3

import signal
import argparse
from formal_bpe.model_exact_dfs import ExactDFSBPE
from formal_bpe.model_exact_dfs_mem import ExactDFSMemBPE
import time
import json
import random

args = argparse.ArgumentParser()
args.add_argument("--merge-count", type=int, default=3)
args.add_argument("--logfile", default="computed/dfs_times.jsonl")
args.add_argument("--line-count", type=int, default=2)
args.add_argument("--seed", type=int, default=0)
args = args.parse_args()
random.seed(args.seed)

times_dyn = []
times_brute = []

CHAR_COUNT = 64

example = " ".join(
    [line[:CHAR_COUNT] for line in
    random.sample(
        open("data/CCrawl.de-en/dev.tok.en", "r").readlines(),
        k=args.line_count
    )]
).replace("\n", "")
print(example)
# print(len(example))
# exit()


def compute_times():
    print("DFS+Memoization")
    time_start = time.time()
    model = ExactDFSMemBPE(fix_overlap=True)
    result_dfsmem = model.fit_greedy(
        example, T=args.merge_count,
    )
    time_dfsmem = time.time() - time_start

    print("DFS")
    time_start = time.time()
    model = ExactDFSBPE(fix_overlap=True)
    result_dfs = model.fit_greedy(
        example, T=args.merge_count,
    )
    time_dfs = time.time() - time_start

    n_dfsmem = len(result_dfsmem)
    n_dfs = len(result_dfs)
    print(result_dfsmem)
    print(result_dfs)

    print("lengths", n_dfsmem, n_dfs)

    print(f"DFS {time_dfs:.1f}s")
    print(f"DFS+Mem {time_dfsmem:.1f}s")
    print(f"Ratio {time_dfsmem/time_dfs:.2f}")

    return {
        "status": "OK",
        "time_dfs": time_dfs,
        "time_dfsmem": time_dfsmem,
        "len_dfs": n_dfs,
        "len_dfsmem": n_dfsmem,
    }


def timeout_compute_times(t):
    def timeout_handler(signum, frame):
        raise Exception()

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    # triger alarm in t seconds seconds
    signal.alarm(t)

    t1 = time.time()
    try:
        return compute_times()
    except:
        t2 = time.time()
        return {"status": f"TIMEOUT ({t2-t1:.0f}s)"}
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# maximum 9999 minutes
result = timeout_compute_times(9999 * 60) | {
    "merge_count": args.merge_count,
    "line_count": args.line_count,
    "seed": args.seed,
}
print(result)

if args.logfile is not None:
    with open(args.logfile, "a") as f:
        f.write(json.dumps(
            result, ensure_ascii=False
        ) + "\n")
