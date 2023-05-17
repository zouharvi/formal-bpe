from collections import defaultdict
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq

class GeneratorDFSMerges:
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

    def yield_all_mergeseq(self, tokens, T):
        stack = [([], tokens)]

        while len(stack) != 0:
            merges, tokens = stack.pop()
            if len(merges) == T:
                continue
            pairs = self.get_word_pair_counts(tokens)

            for pair in pairs:
                merges_new = merges + [pair]
                tokens_new = self.apply_merge_slow(tokens, pair)
                stack.append((merges_new, tokens_new))
                
                # yield all "final" seqs
                if len(merges_new) == T:
                    yield (merges_new, tokens_new)
        return
        output = [debug_flat_seq(x) for x in tokens_best]
        return output, merges_best