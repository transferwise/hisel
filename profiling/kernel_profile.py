import pstats
from pstats import SortKey
import cProfile
from tests.kernel_test import KernelTest  # NOQA


def main():

    cProfile.run('KernelTest().test_rbf()',
                 'rbf_kernel_profile')
    p = pstats.Stats('rbf_kernel_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/kernels.py:', 50)

    cProfile.run('KernelTest().test_delta()',
                 'delta_kernel_profile')
    p = pstats.Stats('delta_kernel_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/kernels.py:', 50)


if __name__ == '__main__':
    main()
