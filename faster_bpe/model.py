
from arsenal.datastructures import LocatorMaxHeap
from collections import defaultdict
from faster_bpe.utils import VERBOSITY, check, pairs_in_list, UniqueList, flat_seq, pretty_seq


class Token:
    # TODO: hash and equality can take time proportional to the size of the tuple.
    # We can fix that with hashconsing.
    __slots__ = ('x', 'prev', 'next')

    def __init__(self, x):
        self.x = x
        self.prev = None
        self.next = None

    def __repr__(self):
        return pretty_seq(self.x)
        return flat_seq(self.x)


class FasterBPE:
    def __init__(self):
        pass

    def top_pair(self):
        return self.heap.peek()[0]

    def __repr__(self):
        return repr(list(self))

    def __iter__(self):
        curr = self.root
        while curr is not None:
            yield curr
            curr = curr.next

    @property
    def xs(self):
        return [t.x for t in self]

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

            new = Token((t.x, t.next.x))
            self.n -= 1

            if old.prev is not None:
                prev = old.prev

                self.heap[prev.x, old.x] = self.pos[
                    prev.x, old.x
                ].remove(prev)
                self.heap[prev.x, new.x] = self.pos[
                    prev.x, new.x
                ].append(t.prev)

            else:
                self.root = new
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

    def fit_greedy(self, xs, T):
        # initialization
        self.n = len(xs)
        root = Token(xs[0])
        ys = [root]
        prev = root
        curr = prev
        for t in range(1, len(xs)):
            curr = Token(xs[t])
            ys.append(curr)
            curr.prev, prev = prev, curr
        next = None
        while curr is not None:
            curr.next = next
            curr, next = curr.prev, curr
        self.root = root

        self.heap = LocatorMaxHeap()
        self.pos = defaultdict(UniqueList)
        for (x, y) in pairs_in_list(ys):
            self.heap[x.x, y.x] = self.pos[x.x, y.x].append(x)

        # actual training
        for t in range(T):
            pair = self.top_pair()
            self.merge(pair)

        return self.get_seq()

    def get_seq(self):
        out = []
        curr = self.root
        while curr is not None:
            out.append(str(curr))
            curr = curr.next
        return [str(x) for x in out]
