import pstats
from pstats import SortKey
import cProfile
from tests.feature_selection_test import FeatSelTest  # NOQA


def main():
    cProfile.run('FeatSelTest().test()',
                 'feature_selection_profile')
    p = pstats.Stats('feature_selection_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/feature_selection.py:', 20)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/categorical.py:', 20)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/select.py:', 50)


if __name__ == '__main__':
    main()
