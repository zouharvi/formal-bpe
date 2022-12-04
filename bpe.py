from arsenal import colors
from arsenal.iterextras import window, count
from collections import Counter, defaultdict
from itertools import product
from IPython.display import display


verbosity = 0
throw = True


class Token:
    def __init__(self, x, i):
        self.i = i
        self.x = x
        self.prev = None
        self.next = None
    def __repr__(self):
#        return f'{self.i}:{self.x}'
        return f'{self.x}'.replace(', ', '').replace("'", '')


class SlowBPE:
    def __init__(self, xs):
        self.xs = xs
        self.c = Counter(window(self.xs, 2))
        self.n = len(self.xs)
        self.pos = defaultdict(list)
        for (x,y) in window(self.xs, 2):
            self.pos[x,y].append(x)
    def merge(self, pair):
        self.xs = merge(self.xs, pair)
        self.c = Counter(window(self.xs, 2))
        self.n = len(self.xs)
        self.pos = defaultdict(list)
        for (x,y) in window(self.xs, 2):
            self.pos[x,y].append(x)
        return self
    def top_pair(self):
        return self.c.most_common()[0][0]


def merge(xs, pair):
    ys = []
    i = 0; N = len(xs)
    while i < N:
        if i < N-1 and (xs[i],xs[i+1]) == pair:
            ys.append(pair)
            i += 2
        else:
            ys.append(xs[i])
            i += 1
    return ys


class FastBPE:

    def __init__(self, xs):
        assert len(xs) > 0
        node = {}
        root = Token(xs[0], 0)
        prev = root
        curr = prev
        for t in range(1, len(xs)):
            curr = Token(xs[t], t)
            curr.prev, prev = prev, curr
        next = None
        while curr is not None:
            curr.next = next
            curr, next = curr.prev, curr
        self.root = root

        self.n = len(xs)
#        c = Counter()
        pos = defaultdict(list)
        for (x,y) in window(self, 2):
            pos[x.x, y.x].append(x)
#            c[x.x, y.x] += 1
#        self.c = c
        self.pos = pos

    @property
    def c(self):
        c = Counter()
        for k,v in self.pos.items():
            c[k] += len(v)
        return c

    def top_pair(self):
        return self.c.most_common()[0][0]

    def reset(self):
        return FastBPE(self.xs)

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

 #       print(colors.yellow % 'merge:', pair, ts)

        for t in ts:
            old = t

#            if (t.x, t.next.x) != pair:
#                print(colors.light.red % 'drop', (t.x, t.next.x))
#                continue
            assert (t.x, t.next.x) == pair, (t.x, t.next.x)

            self.n -= 1

            if t in blocklist:
                if verbosity > 0: print('blocked:', t)
                continue

            blocklist.add(t.next)

            new = Token((t.x, t.next.x), (t.i, t.next.i))
            self.n += 1

            if old.prev is not None:
                prev = old.prev

                try:
                    self.pos[prev.x, old.x].remove(prev)
                except ValueError:
                    pass
                self.pos[prev.x, new.x].append(t.prev)

            else:
                self.root = new
                if verbosity > 0: print('update ROOT')

            if old.next.next is not None:
                after = old.next.next

                try:
                    self.pos[old.next.x, after.x].remove(old.next)
                except ValueError:
                    pass

                self.pos[new.x, after.x].append(new)

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

    def fresh_pos(self):
        pos = defaultdict(list)
        for (x,y) in window(self, 2):
            pos[x.x, y.x].append(x)
        return pos

    def to_graph(self):
        from dyna.graphs import LabeledGraph
        g = LabeledGraph()
        for x in self:
            if x.prev is not None:
                g.add_edge(x, '+', x.prev)
            if x.next is not None:
                g.add_edge(x, '-', x.next)
        return g


def check(have, want):
    if have == want:
        if verbosity > 0: print(colors.ok, have)
    else:
        if verbosity > 0 or not throw:
            print(colors.fail)
            print('  have:', have)
            print('  want:', want)
        if throw:
            raise AssertionError(f'\n\n  want: {want}\n  have: {have}\n')


def test_correctness():
    for xs in [
        'a',
        'abc',
        'aaaaa',
        'eaaaabdaaabc',
        'aaaabdaaab',
    ]:
        _test(xs)

    for xs in product('ab', repeat=8):
        _test(xs)

    print(colors.ok)


def _test(xs):

    if verbosity > 0:
        print()
        print(colors.light.yellow % colors.line(80))
        print(colors.light.yellow % f'# {xs}')

    fast = FastBPE(xs)
    slow = SlowBPE(xs)

    for t in count():

        if len(fast.xs) <= 1: break

        #fast = fast.reset()

        pair, _ = fast.c.most_common()[0]

        if verbosity > 0:
            print()
            print(t, pair, fast.c[pair], fast.pos[pair])

        fast.merge(pair)

        slow.merge(pair)

        check(fast.xs, slow.xs)
#        check(fast.n, slow.n)

        check({k:v for k,v in fast.pos.items() if len(v)},
              {k:v for k,v in fast.fresh_pos().items() if len(v)})

        check({k:v for k,v in fast.c.items() if v != 0},
              {k:v for k,v in slow.c.items() if v != 0})

        #print(fast.c - slow.c)
        #print(self.c - fast.c)


def test_benchmark():
    xs = open('/home/timv/Downloads/wikitext-2-v1/wikitext-2/wiki.train.tokens').read()[:10000]

    T = timers()

    M = 1000
    with T['fast']:
        fast = FastBPE(xs)
        for t in range(M):
            pair = fast.top_pair()
            fast.merge(pair)

    with T['slow']:
        slow = SlowBPE(xs)
        for t in range(M):
            pair = slow.top_pair()
            slow.merge(pair)

    T.compare()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
