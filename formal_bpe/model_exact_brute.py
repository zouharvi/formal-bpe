from collections import defaultdict
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq
from typing import Dict, List, Tuple
from rich.progress import track

class ExactBruteBPE:
    def __init__(self, fix_overlap=False):
        self.fix_overlap = fix_overlap
        if fix_overlap:
            self.get_word_pair_counts = self.get_word_pair_counts_fix_overlap


    @staticmethod
    def apply_merge_multiple(token, pair):
        ys_word = []
        i = 0
        N = len(token)
        while i < N:
            # we found a possible merge!
            if i < N - 1 and (token[i], token[i + 1]) == pair:
                # oh no the next one is also a possilbe merge
                if i < N -  2 and (token[i+1], token[i + 2]) == pair:
                    # merge now
                    results_1 = [ys_word + [pair] + x for x in  ExactBruteBPE.apply_merge_multiple(token[i+2:], pair)]
                    # don't merge now
                    results_2 = [ys_word + [token[i]] + x for x in  ExactBruteBPE.apply_merge_multiple(token[i+1:], pair)]
                    return results_1 + results_2
                else:
                    ys_word.append(pair)
                    i += 2
            else:
                # do not apply merge
                ys_word.append(token[i])
                i += 1
        return [ys_word]

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
        return [ys_word]

    @staticmethod
    def get_word_pair_counts(tokens_freqs):
        pairs = defaultdict(int)
        for token, token_freq in tokens_freqs.values():
            for (x, y) in pairs_in_list(token):
                pairs[x, y] += token_freq

        return pairs

    @staticmethod
    def get_word_pair_counts_fix_overlap(tokens):
        pairs = defaultdict(int)
        prev_pair = None
        for (x, y) in pairs_in_list(tokens):
            # increment only if the prev suffix does not match our prefix
            # otherwise wrong estimate on `aaa`
            if (x,y) != prev_pair:
                pairs[x, y] += 1
                prev_pair = (x, y)
            else:
                # make sure to clear it so that for `aaaa` we count it twice
                prev_pair = None

        return pairs

    @staticmethod
    def top_pair(pairs):
        return max(pairs, key=pairs.__getitem__)

    def fit_greedy(self, tokens, T):

        outputs = []
        self.model = ExactBruteBPE(self.fix_overlap)

        pairs = self.get_word_pair_counts(tokens)
        if T == 0 or len(pairs) == 0:
            return [debug_flat_seq(x) for x in tokens]

        # if len(pairs) == 0: ???
        for pair, pair_freq in pairs.items():
            # TODO: resolve the other indecision
            # UPDATE: that one is implicitly fixed by swapping the operands
            tokens_merged = self.apply_merge_multiple(tokens, pair)
            for tokens_merged_option in tokens_merged:
                outputs.append(self.model.fit_greedy(tokens_merged_option, T-1))

        # this mutates tokens_freqs

        output = min(outputs, key=len)
        output = [debug_flat_seq(x) for x in output]
        return output