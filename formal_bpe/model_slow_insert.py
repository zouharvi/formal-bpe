from collections import defaultdict
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq
from typing import Dict, List, Tuple
from rich.progress import track


class SlowInsertBPE:
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
            token = SlowInsertBPE.apply_merge_slow(token, pair)
        return token

    @staticmethod
    def get_word_pair_counts(tokens_freqs):
        raise NotImplementedError
        pairs = defaultdict(int)
        for token, token_freq in tokens_freqs.values():
            for (x, y) in pairs_in_list(token):
                pairs[x, y] += token_freq

        return pairs

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
        base_length = len(self.encode(token, merge_pairs+[pair]))
        best_merge_pairs = merge_pairs+ [pair]

        for i in range(len(merge_pairs)):
            new_merge_pairs = merge_pairs[:i] + [pair]+merge_pairs[i:]
            new_length = len(self.encode(token, new_merge_pairs))
            if new_length < base_length:
                print("HIT")
                best_merge_pairs = new_merge_pairs

        return best_merge_pairs, self.encode(token, best_merge_pairs)

    @staticmethod
    def top_pair(pairs):
        return max(pairs, key=pairs.__getitem__)

    def fit_greedy(self, tokens, T, debug_output=False, progress_bar=False, indecision_output=False):
        indecision = False
        merge_pairs = []
        tokens_raw = list(tokens)

        iterator = track(range(T)) if progress_bar else range(T)
        for t in iterator:
            pairs = self.get_word_pair_counts(tokens)
            if len(pairs) == 0:
                break

            pair = self.top_pair(pairs)

            merge_pairs, tokens = self.insert_best(merge_pairs, pair, tokens_raw)

        tokens = self.encode(tokens, merge_pairs)

        if debug_output:
            flattener = debug_flat_seq
        else:
            flattener = flat_seq

        output = [flattener(x) for x in tokens]

        if indecision_output:
            return output, indecision
        else:
            return output
