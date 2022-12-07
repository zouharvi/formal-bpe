import numpy as np
import pylab as pl
import random
from itertools import product
from time import time
from arsenal import colors, timers, timeit
from model import FasterBPE, SlowBPE

from utils import VERBOSITY, check

VERBOSITY = 0
def test_correctness():
    for xs in [
        'a',
        'abc',
        'aaaaa',
        'eaaaabdaaabc',
        'aaaabdaaab',
    ]:
        _test_correctness(xs)

    for xs in product('ab', repeat=8):
        _test_correctness(xs)

    print(colors.ok)


def _test_correctness(xs):

    if VERBOSITY > 0:
        print()
        print(colors.light.yellow % colors.line(80))
        print(colors.light.yellow % f'# {xs}')

    fast = FasterBPE(xs)
    slow = SlowBPE(xs)

    for t in range(len(xs)):

        if len(fast.xs) <= 1:
            break

        check(fast.pos[fast.top_pair()].n, len(slow.pos[slow.top_pair()]))

        pair = random.choice(list(fast.pos))

        if VERBOSITY > 0:
            print()
            print(t, pair, fast.pos[pair].n, fast.pos[pair])

        fast.merge(pair)
        slow.merge(pair)

        check(fast.xs, slow.xs)
        check(fast.n, slow.n)

        fast.check_pos()


def test_benchmark():

    corpus = open("data/CCrawl.de-en/train.tok.all").read()
    T = timers()
    M = 2000

#    for N in iterview(np.linspace(2**8, 2**15, 10).astype(int)):
    for i in range(8, 16):
        N = 2**i
        print(N, int(np.log2(N)))

        xs = corpus[:N]

        with T['fast'](N=N):
            fast = FasterBPE(xs)
            for _ in range(M):
                if fast.n <= 1:
                    break
                fast.merge(fast.top_pair())

        with T['slow'](N=N):
            slow = SlowBPE(xs)
            for _ in range(M):
                if slow.n <= 1:
                    break
                slow.merge(slow.top_pair())

    #check(fast.n, slow.n)

    T.compare()
    T.plot_feature('N', show_curve=True)
    pl.show()


def test_speed():
    corpus = open("data/CCrawl.de-en/train.tok.all").read()

    xs = corpus  # [:2**20]

    #M = 1000
    M = len(xs)

    print('corpus size', len(xs))

    with timeit('init'):
        fast = FasterBPE(xs)

    print('started:', fast.n)
    start_size = fast.n

    with timeit('main'):
        last_update = time()
        for m in range(M):
            if time() - last_update > 1:
                last_update = time()
                print(
                    f'compressed {m}: {start_size} {colors.rightarrow} {fast.n} ({start_size/fast.n:.2}x smaller)'
                )
                print('current count:', fast.pos[fast.top_pair()].n)
            if fast.n <= 100:
                break
            if fast.pos[fast.top_pair()].n <= 100:
                break
            fast.merge(fast.top_pair())

    print(
        f'compressed: {start_size} {colors.rightarrow} {fast.n} ({start_size/fast.n:.2}x smaller)'
    )


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
