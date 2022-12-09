from collections import defaultdict
from faster_bpe.utils import pairs_in_list, flat_seq, pretty_seq


class SlowBPE:
    def __init__(self, fix_overlap=False, tokenize=False):
        if fix_overlap:
            self.get_pair_counts = self.get_pair_counts_fix_overlap

        self.tokenize = tokenize

    @staticmethod
    def apply_merge_slow(xs, pair):
        ys = []
        for xs_word in xs:
            ys_word = []
            i = 0
            N = len(xs_word)
            while i < N:
                if i < N - 1 and (xs_word[i], xs_word[i + 1]) == pair:
                    ys_word.append(pair)
                    i += 2
                else:
                    ys_word.append(xs_word[i])
                    i += 1
            ys.append(ys_word)
        return ys

    @staticmethod
    def get_pair_counts(xs):
        pairs = defaultdict(int)
        for xs_word in xs:
            for (x, y) in pairs_in_list(xs_word):
                pairs[x, y] += 1

        return pairs

    @staticmethod
    def get_pair_counts_fix_overlap(xs):
        pairs = defaultdict(int)
        prev_pair = None
        for (x, y) in pairs_in_list(xs):
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

    def fit_greedy(self, xs, T):
        # treat everything as one word
        if not self.tokenize:
            xs = [xs.replace(" ", "▁")]
        else:
            # TODO: this is suboptimal because we're computing the same word twice
            # instead of muiltiplying the result by the frequency 
            xs = xs.replace(" ", " ▁").split(" ")

        for t in range(T):
            pairs = self.get_pair_counts(xs)
            pair = self.top_pair(pairs)
            xs = self.apply_merge_slow(xs, pair)

        return [pretty_seq(x) for xs_word in xs for x in xs_word]