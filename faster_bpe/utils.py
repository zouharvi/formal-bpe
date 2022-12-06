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

