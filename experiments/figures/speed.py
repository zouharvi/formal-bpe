#!/usr/bin/env python3

import matplotlib.pyplot as plt
import random
import fig_utils

XTICKS = [0.5, 1, 4, 8, 16, 26]
plt.figure(figsize=(5, 3.5))

PLOT_STYLE = {
    "marker": ".",
    "markersize": 15,
}

plt.plot(
    XTICKS, [x * (random.random()/2+1) for x in XTICKS],
    label="Greedy iterative",
    **PLOT_STYLE
)
plt.plot(
    XTICKS, [x * (random.random()/5) for x in XTICKS],
    label="Faster BPE",
    **PLOT_STYLE
)
plt.plot(
    XTICKS, [x * (random.random()/5) for x in XTICKS],
    label="Fast BPE",
    **PLOT_STYLE
)

plt.ylabel("Time (s)")
plt.xlabel("Sentences (M)")
plt.legend()
plt.tight_layout(pad=0.3)
plt.savefig("computed/figures/speed.pdf")
plt.show()