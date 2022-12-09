from collections import defaultdict
from faster_bpe.utils import pairs_in_list, flat_seq, pretty_seq


class SlowBPE:
    def __init__(self):
        pass

    @staticmethod
    def apply_merge_slow(xs, pair):
        ys = []
        i = 0
        N = len(xs)
        while i < N:
            if i < N - 1 and (xs[i], xs[i + 1]) == pair:
                ys.append(pair)
                i += 2
            else:
                ys.append(xs[i])
                i += 1
        return ys

    @staticmethod
    def get_pair_counts(xs):
        pairs = defaultdict(int)
        prev_suffix = None
        for (x, y) in pairs_in_list(xs):
            # increment only if the prev suffix does not match our prefix
            # otherwise wrong estimate on `aaa`
            if x != prev_suffix:
                pairs[x, y] += 1
                prev_suffix = y
            else:
                # make sure to clear it so that for `aaaa` we count it twice
                prev_suffix = None

        return pairs

    @staticmethod
    def top_pair(pairs):
        return max(pairs, key=pairs.__getitem__)

    def fit_greedy(self, xs, T):
        for t in range(T):
            pairs = self.get_pair_counts(xs)
            pair = self.top_pair(pairs)
            xs = self.apply_merge_slow(xs, pair)

        return [pretty_seq(x) for x in xs]
