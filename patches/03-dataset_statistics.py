#!/usr/bin/env python3

import numpy as np
import rich.progress

data = open("data/CCrawl.de-en/train.tok.all").readlines()
data = [x.rstrip("\n") for x in rich.progress.track(data)]

sent_char_lengths = [len(x) for x in data]
print("Total length:", sum(sent_char_lengths))
sent_word_lengths = [len(x.split(" ")) + 1 for x in data]
print("Total words:", sum(sent_word_lengths))
unique_words = {w for x in data for w in x.split(" ")}
print("Unique words:", len(unique_words))
unique_words_length = sum([len(w) for w in unique_words])
print("Unique words length:", unique_words_length)
print("Average sentence length (words):", f"{np.average(sent_word_lengths):.0f}")
