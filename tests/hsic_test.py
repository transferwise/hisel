import unittest
import numpy as np
from scipy.stats import special_ortho_group
from hisel import hsic
from hisel.kernels import KernelType
import datetime


QUICK_TEST = True
SERIAL_V_PARALLEL = not QUICK_TEST


class HSICTest(unittest.TestCase):

    def test_search_twocomp_small(self):
        d = 10
        n = 1000
        im_ratio = .4
        idx = np.random.choice(d, replace=False, size=2)
        self._test_search_two_components_classif(
            d, n, idx, num_permutations=1, im_ratio=im_ratio)

    @unittest.skipIf(QUICK_TEST, 'Skipping as QUICK_TEST = True')
    def test_search_twocomp_large(self):
        d = 50
        n = 500
        idx = np.random.choice(d, replace=False, size=2)
        im_ratio = .3
        self._test_search_two_components_classif(
            d, n, idx, num_permutations=1, im_ratio=im_ratio)

    def _test_search_two_components_classif(self, d, n, idx, num_permutations=3, im_ratio=.4):
        print('\n\n##############################################')
        print('# Test HSIC search with classification target')
        print('##############################################\n')
        print(f'Total number of features: {d}')
        print(f'Expected number of relevant features: {len(idx)}')
        print(f'Number of samples: {n}')
        x = np.random.uniform(size=(n, d))
        y = np.array(x[:, idx[0]] > x[:, idx[1]],
                     dtype=int).reshape(n, 1)
        sel = hsic.search(
            x, y, num_permutations=num_permutations, im_ratio=im_ratio)
        selected = set(sel)
        expected = set(idx)
        err = f'expected: {sorted(expected)}\n'
        err += f'selected: {sorted(selected)}\n'
        print(err)
        self.assertTrue(
            expected.issubset(selected) or (
                len(selected) >= len(expected) // 2 and
                selected.issubset(expected)
            ),
            msg=err,
        )

    def test_search_rbf_classif(self):
        dx = 16
        n = 500
        n_rel = 8
        xkerneltype = KernelType.RBF
        num_permutations = 4
        im_ratio = .1
        u = np.random.randint(low=-4, high=4, size=(1, n_rel))
        a_ = np.concatenate((
            u,
            np.zeros(shape=(1, dx - n_rel), dtype=int)
        ),
            axis=1
        )
        a = np.expand_dims(np.random.permutation(
            a_.T).T, axis=0)
        self._test_search_classif(
            dx, n, a, xkerneltype, num_permutations, im_ratio)

    @unittest.skip('Delta kernels are not accurate')
    def test_search_delta_classif(self):
        dx = 16
        n = 500
        n_rel = 3
        xkerneltype = KernelType.DELTA
        num_permutations = 10
        im_ratio = 1.
        u = np.ones(shape=(1, n_rel))
        a_ = np.concatenate((
            u,
            np.zeros(shape=(1, dx - n_rel), dtype=int)
        ),
            axis=1
        )
        a = np.expand_dims(np.random.permutation(
            a_.T).T, axis=0)
        self._test_search_classif(
            dx, n, a, xkerneltype, num_permutations, im_ratio)

    def _test_search_classif(self, d, n, a, xkerneltype: KernelType = KernelType.RBF, num_permutations=3, im_ratio=.4):
        print('\n\n##############################################')
        print('# Test HSIC search with classification target')
        print('##############################################\n')
        print(f'Total number of features: {d}')
        print(f'Number of samples: {n}')
        print(f'xkerneltype: {xkerneltype}')
        if xkerneltype == KernelType.DELTA:
            ms = np.random.randint(low=2, high=3+2*d, size=d)
            xs = np.concatenate([
                np.random.randint(m, size=(n, 1, 1))
                for m in ms
            ], axis=1)
        else:
            xs = np.random.uniform(size=(n, d, 1))
        yvals = a @ xs
        x = xs[:, :, 0]
        yval = yvals[:, :, 0]
        y = (
            np.array(yval > np.quantile(yval, .25), dtype=int) +
            np.array(yval > np.quantile(yval, .5), dtype=int) +
            np.array(yval > np.quantile(yval, .75), dtype=int)
        )
        expected_ = np.where(np.linalg.norm(a[0, :, :], axis=0) > 0)[0]
        expected = set(expected_)
        print(f'Expected number of relevant features: {len(expected)}')
        sel = hsic.search(
            x, y, num_permutations=num_permutations, max_iter=4, im_ratio=im_ratio)
        selected = set(sel)
        err = f'expected: {sorted(expected)}\n'
        err += f'selected: {sorted(selected)}\n'
        print(err)
        self.assertTrue(
            expected.issubset(selected) or (
                len(selected) >= len(expected) // 2 and
                selected.issubset(expected)
            ) or (
                len(selected.symmetric_difference(expected)) < min(
                    len(selected), len(expected)) // 2
            ),
            msg=err,
        )

    def test_search_regression_small(self):
        dx = 10
        n = 1000
        n_rel = dx // 3
        dy = n_rel
        num_permutations = 4
        im_ratio = .35
        u = special_ortho_group.rvs(n_rel)[:dy, :]
        a_ = np.concatenate((
            u,
            np.zeros(shape=(dy, dx - n_rel))
        ),
            axis=1
        )
        a = np.expand_dims(np.random.permutation(
            a_.T).T, axis=0)
        self._test_search_regressor(
            dx, n, a, num_permutations, im_ratio, parallel=SERIAL_V_PARALLEL)

    @unittest.skipIf(QUICK_TEST, 'Skipping as QUICK_TEST = True')
    def test_search_regression_large(self):
        dx = 40
        n = 500
        n_rel = 10
        dy = 6
        num_permutations = 1
        im_ratio = .3
        u = special_ortho_group.rvs(n_rel)[:dy, :]
        a_ = np.concatenate((
            u,
            np.zeros(shape=(dy, dx - n_rel))
        ),
            axis=1
        )
        a = np.expand_dims(np.random.permutation(
            a_.T).T, axis=0)
        self._test_search_regressor(
            dx, n, a, num_permutations, im_ratio, parallel=SERIAL_V_PARALLEL)

    def _test_search_regressor(self, d, n, a, num_permutations=10, im_ratio=.5, parallel: bool = False):
        print('\n\n##############################################')
        print('# Test HSIC search with regression target')
        print('##############################################\n')
        print(f'Total number of features: {d}')
        print(f'Dimensionality of target: {a.shape[1]}')
        print(f'Number of samples: {n}')
        xs = np.random.uniform(size=(n, d, 1))
        ys = a @ xs
        x = xs[:, :, 0]
        y = ys[:, :, 0]
        expected_ = np.where(np.linalg.norm(a[0, :, :], axis=0) > 0)[0]
        expected = set(expected_)
        t0 = datetime.datetime.now()
        sel = hsic.search(
            x, y,
            num_permutations=num_permutations,
            im_ratio=im_ratio,
            parallel=False,
        )
        t1 = datetime.datetime.now()
        dt = t1 - t0
        serruntime = dt.seconds + 1e-6 * dt.microseconds
        selected = set(sel)
        err = f'expected: {sorted(expected)}\n'
        err += f'selected: {sorted(selected)}\n'
        print(err)
        print(f'Serial run time: {serruntime} seconds')
        self.assertTrue(
            expected.issubset(selected) or (
                len(selected) >= len(expected) // 2 and
                selected.issubset(expected)
            ) or (
                len(selected.symmetric_difference(expected)) < min(
                    len(selected), len(expected)) // 2
            ),
            msg=err,
        )
        if parallel:
            t0 = datetime.datetime.now()
            sel = hsic.search(
                x, y,
                num_permutations=num_permutations,
                im_ratio=im_ratio,
                parallel=True,
            )
            t1 = datetime.datetime.now()
            dt = t1 - t0
            parruntime = dt.seconds + 1e-6 * dt.microseconds
            selected = set(sel)
            err = f'expected: {sorted(expected)}\n'
            err += f'Parallel selected: {sorted(selected)}\n'
            print(err)
            print(f'Parallel run time: {parruntime} seconds')
            self.assertTrue(
                expected.issubset(selected) or (
                    len(selected) >= len(expected) // 2 and
                    selected.issubset(expected)
                ) or (
                    len(selected.symmetric_difference(expected)) < min(
                        len(selected), len(expected)) // 2
                ),
                msg=err,
            )


if __name__ == '__main__':
    unittest.main()
