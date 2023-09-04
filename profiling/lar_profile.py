import pstats
from pstats import SortKey
import cProfile
from tests.lar_test import TestLar  # NOQA


def main():
    cProfile.run('TestLar().test_with_gaussian_x()', 'lar_profile')
    p = pstats.Stats('lar_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats('[n\/]lar.py:', 15)


if __name__ == '__main__':
    main()
