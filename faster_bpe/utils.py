from arsenal import colors
VERBOSITY=0
THROW=True

def check(have, want):
    if have == want:
        if VERBOSITY > 0:
            print(colors.ok, have)
    else:
        if VERBOSITY > 0 or not THROW:
            print(colors.fail)
            print('  have:', have)
            print('  want:', want)
        if THROW:
            raise AssertionError(f'\n\n  want: {want}\n  have: {have}\n')


def pairs_in_list(xs):
    return zip(xs, xs[1:])

class UniqueList:
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
        return f'UniqueList({self.xs})'


def pretty_seq(x):
    if type(x) is list:
        return "".join([pretty_seq(i) for i in x])
    if type(x) is str:
        return x
    if type(x) is tuple:
        return f"({pretty_seq(x[0])}{pretty_seq(x[1])})"

def flat_seq(x):
    if type(x) is list:
        return "".join([flat_seq(i) for i in x])
    if type(x) is str:
        return x
    if type(x) is tuple:
        return f"{flat_seq(x[0])}{flat_seq(x[1])}"