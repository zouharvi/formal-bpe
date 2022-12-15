from collections import defaultdict
import itertools
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq
from typing import Dict, List, Tuple
from rich.progress import track


class SlowExactPermBPE:
    def __init__(self, fix_overlap=False, tokenize=False):
        if fix_overlap:
            self.get_word_pair_counts = self.get_word_pair_counts_fix_overlap

        self.tokenize = tokenize

    @staticmethod
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

    @staticmethod
    def encode(token, pairs):
        for pair in pairs:
            token = SlowExactPermBPE.apply_merge_slow(token, pair)
        return token

    @staticmethod
    def get_word_pair_counts(tokens_freqs):
        raise NotImplementedError

    @staticmethod
    def rerank_pairs(pairs, merge_operations, token_raw):
        new_pairs = {}
        for pair in pairs:
            for new_sequence in itertools.permutations(merge_operations + [pair]):
                new_score = SlowExactPermBPE.encode(token_raw, new_sequence)
                if pair not in new_pairs or len(new_pairs[pair]) > len(new_score):
                    new_pairs[pair] = new_score
        return new_pairs

    @staticmethod
    def get_word_pair_counts_fix_overlap(token):
        pairs = defaultdict(int)
        prev_pair = None
        for (x, y) in pairs_in_list(token):
            # increment only if the prev suffix does not match our prefix
            # otherwise wrong estimate on `aaa`
            if (x, y) != prev_pair:
                pairs[x, y] += 1
                prev_pair = (x, y)
            else:
                # make sure to clear it so that for `aaaa` we count it twice
                prev_pair = None

        return pairs

    def insert_best(self, merge_pairs, pair, token):
        # only for comparison, is not actually an edge case
        base_length = len(self.encode(token, merge_pairs + [pair]))
        best_merge_pairs = merge_pairs + [pair]

        for i in range(len(merge_pairs)):
            new_merge_pairs = merge_pairs[:i] + [pair] + merge_pairs[i:]
            new_length = len(self.encode(token, new_merge_pairs))
            if new_length < base_length:
                print("HIT")
                best_merge_pairs = new_merge_pairs

        return best_merge_pairs, self.encode(token, best_merge_pairs)

    @staticmethod
    def top_pair(pairs):
        return min(pairs.items(), key=lambda x: len(x[1]))

    def fit_greedy(self, tokens, T):
        print()
        token_raw = list(tokens)
        merge_operations = []
        for t in range(T):
            pairs = self.get_word_pair_counts(tokens)
            if len(pairs) == 0:
                break
            print(pairs)
            pairs = self.rerank_pairs(pairs, merge_operations, token_raw)
            print({k:len(v) for k, v in pairs.items()})
            pair = self.top_pair(pairs)
            tokens = pair[1]
            merge_operations.append(pair[0])

        outputs = [
            SlowExactPermBPE.encode(token_raw, new_sequence)
            for new_sequence in itertools.permutations(merge_operations)
        ]

        output = min(outputs, key=len)
        output = [debug_flat_seq(x) for x in output]
        print(merge_operations)
        return output
