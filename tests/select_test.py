from typing import Optional
import unittest
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import Selector, FeatureType

use_pyhsiclasso = True
try:
    import pyHSICLasso
except (ModuleNotFoundError, ImportError):
    use_pyhsiclasso = False

try:
    import torch
    SKIP_CUDA = False if torch.cuda.is_available() else True
except (ModuleNotFoundError, ImportError):
    SKIP_CUDA = True

QUICK_TEST = False
SKIP_CUDA = True if QUICK_TEST else SKIP_CUDA
use_pyhsiclasso = False if QUICK_TEST else use_pyhsiclasso


def pyhsiclasso(x, y, yfeattype: FeatureType, n_features: int, batch_size=500):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    if yfeattype == FeatureType.DISCR:
        lasso.classification(n_features, B=batch_size)
    else:
        lasso.regression(n_features, B=batch_size)
    return lasso.A


class SelectorTest(unittest.TestCase):
    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_no_noise(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=False)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_with_noise(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=True)

    def test_regression_no_noise_with_transform(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=False, apply_transform=True)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_regression_with_noise_with_transform(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=True, apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=False, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=True, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise_with_transform(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype,
                             add_noise=False, device='cuda', apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise_with_transform(self):
        yfeattype = FeatureType.CONT
        self._test_selection(yfeattype, add_noise=True, device='cuda',
                             apply_transform=True)

    def test_classification_no_noise(self):
        yfeattype = FeatureType.DISCR
        self._test_selection(yfeattype, add_noise=False)

    @unittest.skipIf(QUICK_TEST, 'Skipping for faster test')
    def test_classification_with_noise(self):
        yfeattype = FeatureType.DISCR
        self._test_selection(yfeattype, add_noise=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_classification_no_noise(self):
        yfeattype = FeatureType.DISCR
        self._test_selection(yfeattype, add_noise=False, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_classification_with_noise(self):
        yfeattype = FeatureType.DISCR
        self._test_selection(yfeattype, add_noise=True, device='cuda')

    def _test_selection(
        self,
        yfeattype: FeatureType,
        add_noise: bool = False,
        apply_transform: bool = False,
        device: Optional[str] = None,
    ):
        print('\n\n\n##############################################################')
        print('Test selection of features in a linear transformation setting')
        print('##############################################################')
        print(f'Feature type of y: {yfeattype}')
        print(f'Apply transform: {apply_transform}')
        print(f'Noisy target: {add_noise}')
        print(f'device: {device}')
        d: int = np.random.randint(low=15, high=25)
        n: int = np.random.randint(low=10000, high=20000)
        n_features: int = d // 3
        features = list(np.random.choice(d, replace=False, size=n_features))
        x: np.ndarray = np.random.uniform(size=(n, d))
        z: np.array = x[:, features]
        if apply_transform or yfeattype == FeatureType.DISCR:
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
        if use_pyhsiclasso:
            print('Using pyHSICLasso for reconciliation purposes')
            pyhsiclasso_selection = pyhsiclasso(
                x, y, yfeattype, n_features, 500)
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
            self.assertEqual(
                set(pyhsiclasso_selection),
                set(features),
                msg=(
                    f'\npyhsiclasso_selection: {pyhsiclasso_selection}'
                    f'\nfeatures: {features}\n\n'
                )
            )

        selector = Selector(
            x, y,
            xfeattype=FeatureType.CONT,
            yfeattype=yfeattype
        )
        selection = selector.select(
            n_features, batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, device=device)
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
        self.assertEqual(
            set(selection),
            set(features),
            msg=(f'Expected features: {features}',
                 f'Selected features: {selection}'
                 )
        )

        if QUICK_TEST:
            return
        # Test autoselection - We do not provide the number of features that should be selected
        autoselection = selector.autoselect(
            batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, threshold=3e-2, device=device)
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
                msg=(f'Expected features: {features}',
                     f'Auto-Selected features: {autoselection}'
                     )
            )
        else:
            self.assertLess(
                abs(len(autoselection) - len(features)),
                2
            )
            self.assertLess(
                abs(len(autoselection) - n_features),
                2
            )
            diff_left = set(features).difference(set(autoselection))
            diff_right = set(autoselection).difference(set(features))
            diff = diff_left.union(diff_right)
            self.assertLess(
                len(diff),
                2,
                msg=(f'Expected features: {features}',
                     f'Auto-Selected features: {autoselection}'
                     )
            )


if __name__ == '__main__':
    unittest.main()
