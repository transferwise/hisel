import pstats
from pstats import SortKey
import cProfile
from tests.categorical_test import CategoricalSearchTest  # NOQA


def main():
    cProfile.run('CategoricalSearchTest().test_search_and_select()',
                 'categorical_profile')
    p = pstats.Stats('categorical_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats('hisel/categorical.py:', 20)


if __name__ == '__main__':
    main()
