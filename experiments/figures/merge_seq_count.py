import matplotlib.pyplot as plt
import json
import fig_utils

COLORS = {
    "merge_3": fig_utils.COLORS[0],
    "merge_4": fig_utils.COLORS[1],
    "merge_5": fig_utils.COLORS[2],
}

for key in ["merge_3", "merge_4", "merge_5"]:
    with open(f"computed/merge_count/{key}.json", "r") as f:
        data = json.load(f)

    print(len(data.keys()))
    print([max(v) for v in data.values()])
    keyname = key.replace("merge_", "V=")
    plt.scatter(
        list(data.keys()),
        [max(v)[1] for v in data.values()],
        label=keyname + " DP",
        marker=".",
        color=COLORS[key],
    )
    plt.scatter(
        list(data.keys()),
        [max(v)[0] for v in data.values()],
        label=keyname + " brute",
        marker="x",
        color=COLORS[key],
    )
plt.ylabel("Number of explored sequences")
plt.xlabel("Example size (alphabet size 3)")
plt.legend()
plt.show()