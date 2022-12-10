#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import random
import fig_utils
import argparse

args = argparse.ArgumentParser()
args.add_argument(
    "-ls", "--logfiles-small", nargs="+",
    default=[
        "computed/time_faster_notok.jsonl",
        "computed/time_faster_tok.jsonl",
        "computed/time_slow_tok.jsonl",
        "computed/time_slow_notok.jsonl",
        "computed/time_sentencepiece.jsonl",

    ]
)
args.add_argument(
    "-ll", "--logfiles-large", nargs="+",
    default=[
        "computed/time_faster_notok.jsonl",
        "computed/time_faster_tok.jsonl",
        "computed/time_sentencepiece.jsonl",
    ]
)
args = args.parse_args()

# TODO: fancier style

COLORS = {
    "faster_notok": fig_utils.COLORS[0],
    "slow_tok": fig_utils.COLORS[1],
    "slow_notok": fig_utils.COLORS[2],
    "sentencepiece": fig_utils.COLORS[3],
    "faster_tok": fig_utils.COLORS[4],
}

data_small = {}
for file in args.logfiles_small:
    name = file.split("/")[-1].split(".")[0].removeprefix("time_")
    data_small[name] = [json.loads(x) for x in open(file, "r").readlines()]

data_large = {}
for file in args.logfiles_large:
    name = file.split("/")[-1].split(".")[0].removeprefix("time_")
    data_large[name] = [json.loads(x) for x in open(file, "r").readlines()]

fig = plt.figure(figsize=(5, 3.5))
ax1, ax2 = fig.subplots(1, 2)

for model, data_local in data_small.items():
    xticks = [x["n_line"] for x in data_local][:8]
    times = [x["time"] for x in data_local][:8]
    ax1.plot(
        xticks, times,
        alpha=0.5,
        color=COLORS[model],
    )
    ax1.scatter(
        xticks, times,
        label=model,
        marker=".",
        s=100,
        alpha=1,
        color=COLORS[model],
    )

for model, data_local in data_large.items():
    print(model)
    xticks = [x["n_line"] for x in data_local][8:]
    times = [x["time"] for x in data_local][8:]
    ax2.plot(
        xticks, times,
        alpha=0.5,
        color=COLORS[model],
    )
    ax2.scatter(
        xticks, times,
        label=model,
        marker=".",
        s=100,
        alpha=1,
        color=COLORS[model],
    )

ax1.set_ylabel("Time (s)")
ax1.set_xlabel("Line count")
ax2.set_xlabel("Line count")

lines = []
labels = []

for ax in fig.axes:
    line_handles, line_labels = ax.get_legend_handles_labels()
    line_zip = [(x, y) for x, y in zip(line_handles, line_labels) if y not in labels]
    # there's a more elegant way to this but I forgot
    line_handles = [x[0] for x in line_zip]
    line_labels = [x[1] for x in line_zip]
    # print(Label)
    lines.extend(line_handles)
    labels.extend(line_labels)

fig.legend(
    lines, labels,
    bbox_to_anchor=(0, 1, 1, 0),
    loc="upper right",
    ncol=3
)

plt.tight_layout(
    rect=(0, 0, 1, 0.83),
    pad=0
)
plt.savefig("computed/figures/speed.pdf")
plt.show()
