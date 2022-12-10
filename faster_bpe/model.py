
from arsenal.datastructures import LocatorMaxHeap
from collections import defaultdict
from faster_bpe.utils import VERBOSITY, check, pairs_in_list, UniqueList, flat_seq, debug_flat_seq
from rich.progress import track

class Token:
    # TODO: hash and equality can take time proportional to the size of the tuple.
    # We can fix that with hashconsing.
    __slots__ = ('x', 'prev', 'next', "line_i")

    def __init__(self, x, line_i):
        self.x = x
        self.prev = None
        self.next = None
        self.line_i = line_i

    def __repr__(self):
        return flat_seq(self.x)
        return debug_flat_seq(self.x)

class FasterBPE:
    def __init__(self, tokenize=False):
        self.tokenize = tokenize

    def top_pair(self):
        if self.tokenize:
            # pop until we get a valid pair
            while True:
                pair = self.heap.peek()[0]
                if " " in pair:
                    self.heap.pop()
                else:
                    return pair
        else:
            return self.heap.peek()[0]

    def merge(self, pair):
        blocklist = set()
        ts = self.pos.pop(pair, ())

        self.heap[pair] = 0

        for t in ts:
            old = t

            assert (t.x, t.next.x) == pair, (t.x, t.next.x)

            if t in blocklist:
                if VERBOSITY > 0:
                    print('blocked:', t)
                continue

            blocklist.add(t.next)

            new = Token((t.x, t.next.x), t.line_i)

            if old.prev is not None:
                prev = old.prev

                self.heap[prev.x, old.x] = self.pos[
                    prev.x, old.x
                ].remove(prev)
                self.heap[prev.x, new.x] = self.pos[
                    prev.x, new.x
                ].append(t.prev)

            else:
                self.roots[t.line_i] = new
                if VERBOSITY > 0:
                    print('update ROOT')

            if old.next.next is not None:
                after = old.next.next

                self.heap[
                    old.next.x, after.x
                ] = self.pos[old.next.x, after.x].remove(old.next)

                self.heap[
                    new.x, after.x
                ] = self.pos[new.x, after.x].append(new)

            if old.prev is not None:
                prev = old.prev
                assert prev.next == old, [prev, prev.next, old]

                prev.next = new
                new.prev = prev

            if old.next.next is not None:
                after = old.next.next

                new.next = after
                after.prev = new

            old.prev = None
            old.next = None

        return self

    def check_pos(self):
        pos = defaultdict(UniqueList)
        for (x, y) in pairs_in_list(list(self)):
            pos[x.x, y.x].append(x)

        check(
            {k: v for k, v in self.pos.items() if len(v)},
            {k: v for k, v in pos.items() if len(v)}
        )

    def fit_greedy(self, tokens, T, progress_bar=False):
        # treat the whole line as one word
        # the tokenization effect takes place only in top_pair
        if self.tokenize:
            tokens = [line.replace(" ", " ▁") for line in tokens.split("\n")]
        else:
            tokens = [line.replace(" ", "▁") for line in tokens.split("\n")]

        tokens = [" " if len(line) == 0 else line for line in tokens]

        # initialization
        self.roots = [Token(line[0], line_i) for line_i, line in enumerate(tokens)]
        ys = []
        for line_i, (line_root, line) in enumerate(zip(self.roots, tokens)):
            ys_line = [line_root]
            prev = line_root
            curr = prev
            for t in range(1, len(line)):
                curr = Token(line[t], line_i)
                ys_line.append(curr)
                curr.prev, prev = prev, curr
            next = None
            while curr is not None:
                curr.next = next
                curr, next = curr.prev, curr
            ys.append(ys_line)

        self.heap = LocatorMaxHeap()
        self.pos = defaultdict(UniqueList)
        for ys_line in ys:
            for (x, y) in pairs_in_list(ys_line):
                self.heap[x.x, y.x] = self.pos[x.x, y.x].append(x)

        # actual training
        iterator = track(range(T)) if progress_bar else range(T)
        for t in iterator:
            pair = self.top_pair()
            self.merge(pair)

        return [self.get_seq(root) for root in self.roots]

    def get_seq(self, root):
        out = []
        curr = root
        while curr is not None:
            out.append(str(curr))
            curr = curr.next
        # filter out single spaces - they are never valid subwords on their own
        return [str(x) for x in out if x != " "]
