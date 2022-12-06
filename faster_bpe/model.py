
from arsenal.datastructures import LocatorMaxHeap
from collections import defaultdict
from utils import VERBOSITY, check

VERBOSITY = 0

def pairs(xs):
    return zip(xs, xs[1:])


# TODO: hash and equality can take time proportional to the size of the tuple.
# We can fix that with hashconsing.
class Token:
    __slots__ = ('x', 'prev', 'next')

    def __init__(self, x):
        self.x = x
        self.prev = None
        self.next = None

    def __repr__(self):
        return f'{self.x}'.replace(', ', '').replace("'", '')


class SlowBPE:
    def __init__(self, xs):
        self.xs = xs
        self.n = len(self.xs)
        self.pos = defaultdict(list)
        for (x, y) in pairs(self.xs):
            self.pos[x, y].append(x)

    def merge(self, pair):
        self.xs = merge(self.xs, pair)
        self.n = len(self.xs)
        self.pos = defaultdict(list)
        for (x, y) in pairs(self.xs):
            self.pos[x, y].append(x)
        return self

    def top_pair(self):
        return max(self.pos, key=lambda pair: len(self.pos[pair]))


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


class MyList:
    def __init__(self):
        self.xs = []
        self.loc = {}
        self.n = 0

    def append(self, x):
        i = len(self.xs)
        assert x not in self.loc
        self.xs.append(x)
        self.loc[x] = i
        self.n += 1
        return self.n

    def remove(self, x):
        if x in self.loc:
            i = self.loc.pop(x)
            self.xs[i] = None
            self.n -= 1
        return self.n

    def __len__(self):
        return self.n

    def __iter__(self):
        for x in self.xs:
            if x is not None:
                yield x

    def __eq__(self, other):
        return list(self) == list(other)

    def __repr__(self):
        return f'MyList({self.xs})'


class FastBPE:

    def __init__(self, xs):
        assert len(xs) > 0
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

        self.n = len(xs)
        self.heap = LocatorMaxHeap()
        self.pos = defaultdict(MyList)
        for (x, y) in pairs(ys):
            self.heap[x.x, y.x] = self.pos[x.x, y.x].append(x)

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
            self.n -= 2
            self.n += 1

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

                self.heap[old.next.x, after.x] = self.pos[old.next.x, after.x].remove(old.next)
                self.heap[new.x, after.x] = self.pos[new.x, after.x].append(new)

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
        pos = defaultdict(MyList)
        for (x, y) in pairs(list(self)):
            pos[x.x, y.x].append(x)

        check(
            {k: v for k, v in self.pos.items() if len(v)},
            {k: v for k, v in pos.items() if len(v)}
        )

