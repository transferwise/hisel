import pstats
from pstats import SortKey
import cProfile
from tests.hsic_test import HSICTest


def main():
    cProfile.run('HSICTest().test_search_regression_small()',
                 'hsic_profile')
    p = pstats.Stats('hsic_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/hsic.py:', 50)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/kernels.py:', 50)


if __name__ == '__main__':
    main()
