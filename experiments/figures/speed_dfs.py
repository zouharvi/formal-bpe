#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MaxNLocator
import numpy as np
import fig_utils
import collections
import argparse

args = argparse.ArgumentParser()
args.add_argument(
    "--logfile",
    default="computed/dfs_times.jsonl",
)
args = args.parse_args()

data_raw = [json.loads(x) for x in open(args.logfile, "r") if len(x) > 1]
data = collections.defaultdict(list)
for line in data_raw:
    data[line["merge_count"]].append(line)

data = sorted(data.items(), key=lambda x: x[0])

values = []
for merge_count, lines in data:
    lines = [line for line in lines if line["status"] == "OK"]
    values.append(
        (
            merge_count,
            np.average([line["time_dfs"] for line in lines]),
            np.average([line["time_dfsmem"] for line in lines]),
        )
    )


def signif(x, p):
    if x > 1:
        return f"{x:.0f}"
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return f"{np.round(x * mags) / mags}$".replace(".0$", "").removesuffix("$")


fig = plt.figure(figsize=(4, 1.8))
ax1 = fig.gca()

xticks = [x[0] for x in values]
yvals =     [x[1] / x[2] for x in values]

ax1.plot(
    xticks, yvals,
    label=f"Ratio",
    alpha=0.5,
    color="black",
)
ax1.scatter(
    xticks, yvals,
    color="gray",
)


for x in values:
    ax1.text(
        x[0], x[1] / x[2] + 2,
        s=f"{signif(x[1], 1)}s\n{signif(x[2], 1)}s",
        ha="center", va="center",
    )


# force integer xticks
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yticks(range(1, 16, 4))
ax1.set_ylim(ax1.get_ylim()[0] - 0.1, ax1.get_ylim()[1] + 3.5)
ax1.set_xlim(ax1.get_xlim()[0] - 0.3, ax1.get_xlim()[1] + 0.3)
ax1.set_xlabel("Merge count")
ax1.set_ylabel("DFS to DFS+Mem\nRuntime ratio")

pos_text_x = 1
pos_text_y = max(yvals)+3

ax1.text(
    x= pos_text_x,
    y= pos_text_y,
    s="DFS time (s)\nDFS+Mem time (s)",
    ha="left", va="top",
    font={"size": 9},
)
ax1.annotate(
    "",
    xy=(4.7, pos_text_y-0.6), xycoords='data',
    xytext=(pos_text_x+1.2, pos_text_y-0.5), textcoords='data',
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="angle3,angleA=5,angleB=-5",
        color="black",
    ),
)
ax1.annotate(
    "",
    xy=(4.8, pos_text_y-2), xycoords='data',
    xytext=(pos_text_x+1.8, pos_text_y-1.9), textcoords='data',
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="angle3,angleA=10,angleB=-10",
        color="black",
    ),
)

# plt.legend()
plt.tight_layout(pad=0.5)
plt.savefig("computed/figures/dfs_speed.pdf")
plt.show()
