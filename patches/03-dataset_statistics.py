#!/usr/bin/env python3

import numpy as np

data = open("data/CCrawl.de-en/train.tok.all").readlines()
data = [x.rstrip("\n") for x in data]

sent_lengths = [len(x.split(" ")) + 1 for x in data]
print("Total words:", sum(sent_lengths))
unique_words = len({w for x in data for w in x.split(" ")})
print("Unique words:", unique_words)
print("Average sentence length (words):", f"{np.average(sent_lengths):.0f}")
