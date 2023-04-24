import unittest
import numpy as np
from hisel import lar
use_pyhsiclasso = True
try:
    from pyHSICLasso import nlars
except (ModuleNotFoundError, ImportError):
    use_pyhsiclasso = False


class TestLar(unittest.TestCase):

    def test_nonneg_beta_no_noise(self):
        n = 1000
        d = 10
        a = 4
        x = np.random.uniform(size=(n, d))
        beta = np.random.permutation(np.vstack(
            [
                np.random.uniform(size=(a, 1)),
                np.zeros((d-a, 1), dtype=np.float32)
            ]
        ))
        y = x @ beta
        self._test(x, y, a, beta)

    def test_no_noise(self):
        n = 1000
        d = 10
        a = 4
        x = np.random.uniform(size=(n, d))
        beta = np.random.permutation(np.vstack(
            [
                np.random.uniform(low=-1., high=1., size=(a, 1)),
                np.zeros((d-a, 1), dtype=np.float32)
            ]
        ))
        y = x @ beta
        self._test(x, y, a, beta)

    def test_nonneg_beta_with_noise(self):
        n = 1000
        d = 10
        a = 4
        x = np.random.uniform(size=(n, d))
        beta = np.random.permutation(np.vstack(
            [
                np.random.uniform(low=0., high=1., size=(a, 1)),
                np.zeros((d-a, 1), dtype=np.float32)
            ]
        ))
        y = x @ beta + np.random.uniform(low=-1e-2, high=1e-2, size=(n, 1))
        self._test(x, y, a, beta)

    def test_big_nonneg_beta_with_noise(self):
        n = 100000
        d = 100
        a = 20
        x = np.random.uniform(size=(n, d))
        beta = np.random.permutation(np.vstack(
            [
                np.random.uniform(low=0., high=1., size=(a, 1)),
                np.zeros((d-a, 1), dtype=np.float32)
            ]
        ))
        y = x @ beta + np.random.uniform(low=-1e-2, high=1e-2, size=(n, 1))
        self._test(x, y, a, beta)

    def _test(self, x, y, a, beta):
        print('\nTesting implementation of least-angle regression in `hisel`')
        active, path = lar.solve(x, y, a)
        if path.ndim == 2 and path.shape[0] >= 1:
            active = sorted(
                active,
                key=lambda a: path[-1, a],
                reverse=True
            )
        nonactive = list(set(range(x.shape[1])).difference(set(active)))
        if use_pyhsiclasso:
            print('Using pyHSICLasso.nlars for reconciliation purposes')
            _, _, a_nlar, _, _, _ = nlars.nlars(x, x.T @ y, a, 3)
            self.assertEqual(
                set(a_nlar),
                set(active),
                msg=('\npyHSICLasso and hisel did not find the same set of features!\n'
                     f'hisel-selected features: {set(active)}\n'
                     f'pyHSICLasso-selected features: {set(a_nlar)}\n'
                     )
            )
            self.assertEqual(
                a_nlar,
                active,
                msg=('\npyHSICLasso and hisel did not agree on the order of the features\n'
                     f'hisel-ordered beta:\n{beta[active]}\n'
                     f'pyHSICLasso-ordered beta:\n{beta[a_nlar]}\n'
                     )
            )
        if np.all(beta >= -1e-12):
            # If any of the entries in beta is negative, the
            # following tests are not guaranteed to pass
            self.assertTrue(
                np.all(beta[active] >= .0),
                msg=('hisel has selected variables with negative beta\n'
                     f'beta:\n{beta}\n'
                     f'selected beta:\n{beta[active]}\n'
                     )
            )
            self.assertTrue(
                np.all(beta[nonactive] <= .0),
                msg=('hisel has not selected variables with positive beta\n'
                     f'beta:\n{beta}\n'
                     f'selected beta:\n{beta[active]}\n'
                     )
            )


if __name__ == '__main__':
    unittest.main()
