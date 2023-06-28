from typing import Optional
import unittest
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import HSICSelector as Selector, FeatureType
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

USE_PYHSICLASSO = True
try:
    import pyHSICLasso
except (ModuleNotFoundError, ImportError):
    USE_PYHSICLASSO = False

try:
    import torch
    SKIP_CUDA = False if torch.cuda.is_available() else True
except (ModuleNotFoundError, ImportError):
    SKIP_CUDA = True

QUICK_TEST = True
SKIP_CUDA = True if QUICK_TEST else SKIP_CUDA
USE_PYHSICLASSO = False if QUICK_TEST else USE_PYHSICLASSO
SKLEARN_RECON = True


def pyhsiclasso(x, y, xfeattype,  yfeattype, n_features: int, batch_size=500):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    discrete_x = False  # xfeattype == FeatureType.DISCR
    if yfeattype == FeatureType.DISCR:
        lasso.classification(n_features, B=batch_size, discrete_x=discrete_x)
    else:
        lasso.regression(n_features, B=batch_size, discrete_x=discrete_x)
    return lasso.A


class SelectorTest(unittest.TestCase):
    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_no_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype, add_noise=False)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype, add_noise=True)

    def test_regression_no_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, apply_transform=True)

    def test_categorical_regression_no_noise_with_transform(self):
        xfeattype = FeatureType.DISCR
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, apply_transform=True)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_with_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=True, apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=True, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, device='cuda', apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype, add_noise=True, device='cuda',
                             apply_transform=True)

    def test_classification_no_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype, add_noise=False)

    def test_categorical_classification_no_noise(self):
        xfeattype = FeatureType.DISCR
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, apply_transform=False)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_classification_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype, add_noise=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_classification_no_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_classification_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype,
                             add_noise=True, device='cuda')

    def _test_selection(
        self,
        xfeattype: FeatureType,
        yfeattype: FeatureType,
        add_noise: bool = False,
        apply_transform: bool = False,
        device: Optional[str] = None,
    ):
        print('\n\n\n##############################################################')
        print('Test selection of features in a linear transformation setting')
        print('##############################################################')
        print(f'Feature type of x: {xfeattype}')
        print(f'Feature type of y: {yfeattype}')
        print(f'Apply transform: {apply_transform}')
        print(f'Noisy target: {add_noise}')
        print(f'device: {device}')
        d: int = np.random.randint(low=15, high=25)
        n: int = np.random.randint(low=5000, high=10000)
        n_features: int = d // 3
        features = list(np.random.choice(d, replace=False, size=n_features))
        x: np.ndarray
        y: np.ndarray
        if xfeattype == FeatureType.DISCR:
            ms = np.random.randint(low=2, high=2*n_features, size=(d,))
            xs = [np.random.randint(m, size=(n, 1)) for m in ms]
            x = np.concatenate(xs, axis=1)
        else:
            x = np.random.uniform(size=(n, d))
        z: np.array = x[:, features]
        if (apply_transform or yfeattype == FeatureType.DISCR):
            tt = np.expand_dims(
                special_ortho_group.rvs(n_features),
                axis=0
            )
            zz = np.expand_dims(z, axis=2)
            u = (tt @ zz)[:, :, 0]
        else:
            u = z
        if add_noise:
            scaler = .01 if yfeattype == FeatureType.DISCR else .1
            u += scaler * np.std(u) * np.random.uniform(size=u.shape)
        if yfeattype == FeatureType.CONT:
            y = u
        elif yfeattype == FeatureType.DISCR:
            y = np.zeros(shape=(n, 1), dtype=int)
            for i in range(1, n_features):
                y += np.asarray(u[:, [i-1]] > u[:, [i]], dtype=int)
        else:
            raise ValueError(yfeattype)
        print(f'Expected features:\n{sorted(features)}\n')
        if USE_PYHSICLASSO and (xfeattype == FeatureType.CONT):
            print('Using pyHSICLasso for reconciliation purposes')
            pyhsiclasso_selection = pyhsiclasso(
                x, y, xfeattype, yfeattype, n_features, 500)
            print(
                f'pyHSICLasso selected features:\n{sorted(pyhsiclasso_selection)}')
            self.assertEqual(
                len(pyhsiclasso_selection),
                len(features),
            )
            self.assertEqual(
                len(pyhsiclasso_selection),
                n_features,
            )
            msg = (
                f'\npyhsiclasso_selection: {sorted(pyhsiclasso_selection)}'
                f'\nfeatures: {sorted(features)}\n\n'
            )
            if not set(pyhsiclasso_selection) == set(features):
                print(
                    f'WARNING: pyHSICLasso did not perform an exact selection:\n{msg}')

        selector = Selector(
            x, y,
            xfeattype=FeatureType.CONT,  # xfeattype,
            yfeattype=yfeattype
        )
        selection = selector.select(
            n_features, batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, device=device)
        print(f'Expected features:\n{sorted(features)}')
        print(
            f'hisel selected features:\n{sorted(selection)}')
        self.assertEqual(
            len(selection),
            len(features),
        )
        self.assertEqual(
            len(selection),
            n_features,
        )
        if SKLEARN_RECON:
            miy = np.sum(y, axis=1)
            discrete_features = xfeattype == FeatureType.DISCR
            if yfeattype == FeatureType.CONT:
                mi = mutual_info_regression(
                    x, miy, discrete_features=discrete_features)
            else:
                mi = mutual_info_classif(
                    x, miy, discrete_features=discrete_features)
            mi /= np.max(mi)
            miargsort = np.argsort(mi)
            mi_selection = miargsort[::-1][:n_features]
            print(f'mi_selection:\n{sorted(mi_selection)}')
            if set(mi_selection) == set(features):
                self.assertEqual(
                    set(selection),
                    set(mi_selection),
                    msg=(f'MI features: {sorted(mi_selection)}\n'
                         f'Selected features: {sorted(selection)}'
                         )
                )
            else:
                print('WARNING: sklearn mi did not select the right features!')
        if QUICK_TEST and (xfeattype == FeatureType.DISCR or yfeattype == FeatureType.DISCR):
            grace = 3
            diff_left = set(selection).difference(set(features))
            diff_right = set(features).difference(set(selection))
            diff = diff_left.union(diff_right)
            self.assertLess(
                len(diff),
                grace,
                msg=(f'Expected features: {sorted(features)}\n'
                     f'Selected features: {sorted(selection)}'
                     )
            )
        else:
            self.assertEqual(
                set(selection),
                set(features),
                msg=(f'Expected features: {sorted(features)}\n'
                     f'Selected features: {sorted(selection)}'
                     )
            )

        if QUICK_TEST:
            return
        # Test autoselection - We do not provide the number of features that should be selected
        autoselected_features = selector.autoselect(
            batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, threshold=3e-2, device=device)
        autoselection = [int(feat.split('f')[-1])
                         for feat in autoselected_features]
        print(
            f'hisel auto-selected features:\n{sorted(autoselection)}')
        if yfeattype == FeatureType.CONT:
            self.assertEqual(
                len(autoselection),
                len(features),
            )
            self.assertEqual(
                len(autoselection),
                n_features,
            )
            self.assertEqual(
                set(autoselection),
                set(features),
                msg=(f'Expected features: {sorted(features)}\n'
                     f'Auto-Selected features: {sorted(autoselection)}'
                     )
            )
        else:
            threshold = 2 + n_features // 6
            self.assertLess(
                abs(len(autoselection) - len(features)),
                threshold
            )
            self.assertLess(
                abs(len(autoselection) - n_features),
                threshold
            )
            diff_left = set(features).difference(set(autoselection))
            diff_right = set(autoselection).difference(set(features))
            diff = diff_left.union(diff_right)
            self.assertLess(
                len(diff),
                threshold,
                msg=(f'Expected features: {features}\n'
                     f'Auto-Selected features: {autoselection}'
                     )
            )


if __name__ == '__main__':
    unittest.main()
