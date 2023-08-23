import pstats
from pstats import SortKey
import cProfile
from tests.select_test import SelectorTest


def selection_on_cpu():
    cProfile.run('SelectorTest().test_regression_no_noise_with_transform()',
                 'select_profile')
    p = pstats.Stats('select_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/select.py:', 50)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/kernels.py:', 50)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'lar.py:', 50)


def selection_on_gpu():
    cProfile.run('SelectorTest().test_cuda_regression_no_noise_with_transform()',
                 'select_profile')
    p = pstats.Stats('select_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/select.py:', 50)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/cudakernels.py:', 50)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'lar.py:', 50)


def main():
    selection_on_cpu()


if __name__ == '__main__':
    main()
