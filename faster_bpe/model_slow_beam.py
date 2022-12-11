from collections import defaultdict, Counter
from faster_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq
from rich.progress import track
from arsenal.iterextras import take

def possible_splits_gen(xs):
    if len(xs) <= 1:
        yield xs
        return

    c = Counter(pairs_in_list(xs))
    for pair in sorted(c, key=c.__getitem__, reverse=True):
        yield SlowBeamBPE.merge(xs, pair)


def merge(xs, pair):
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


def pairs(xs):
    return zip(xs, xs[1:])

class SlowBeamBPE:
    def __init__(self):
        pass

    @staticmethod
    def top_pair(pairs):
        return max(pairs, key=pairs.__getitem__)

    @staticmethod
    def merge(xs, pair):
        xs = merge(xs, pair)
        pos = defaultdict(list)
        for (x, y) in pairs(xs):
            pos[x, y].append(x)
        return xs

    def fit_beam_search(self, xs, T, B):
        beam = [xs]
        for t in range(T):
            bs = beam + [ys for xs in beam for ys in take(B, possible_splits_gen(xs))]
            beam = sorted(bs, key=len)[:B]
        assert beam[0] == min(beam, key=len)
        return [debug_flat_seq(x) for x in beam[0]]