import pstats
from pstats import SortKey
import cProfile
import numpy as np
from hisel import lar
from scipy.stats import multivariate_normal
from tests.lar_test import TestLar


def main():
    cProfile.run('TestLar().test_with_gaussian_x()', 'lar_profile')
    p = pstats.Stats('lar_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats('[n\/]lar.py:', 15)


if __name__ == '__main__':
    main()
