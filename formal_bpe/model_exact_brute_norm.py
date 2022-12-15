from collections import defaultdict
import copy
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq

class ExactBruteNormBPE:
    def __init__(self, fix_overlap=False):
        if fix_overlap:
            self.get_word_pair_counts = self.get_word_pair_counts_fix_overlap

        self.explored_seq = set()

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
        if T == 0:
            return [debug_flat_seq(x) for x in tokens]

        outputs = []

        pairs = self.get_word_pair_counts(tokens)
        for pair, pair_freq in pairs.items():
            tokens_new = self.apply_merge_slow(tokens, pair)
            outputs.append(self.fit_greedy(tokens_new, T-1))
        else:
            outputs.append(tokens)

        # this mutates tokens_freqs
        output = min(outputs, key=len)
        output = [debug_flat_seq(x) for x in output]
        return output