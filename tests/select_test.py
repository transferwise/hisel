from typing import Optional
import unittest
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import Selector

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


def pyhsiclasso(x, y, n_features: int, batch_size=500):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    lasso.regression(n_features, B=batch_size)
    return lasso.A


class SelectorTest(unittest.TestCase):
    def test_selection_no_noise(self):
        self._test_selection(add_noise=False)

    def test_selection_with_noise(self):
        self._test_selection(add_noise=True)

    def test_selection_no_noise_with_transform(self):
        self._test_selection(add_noise=False, apply_transform=True)

    def test_selection_with_noise_with_transform(self):
        self._test_selection(add_noise=True, apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_selection_no_noise(self):
        self._test_selection(add_noise=False, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_selection_with_noise(self):
        self._test_selection(add_noise=True, device='cuda')

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_selection_no_noise_with_transform(self):
        self._test_selection(
            add_noise=False, device='cuda', apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_selection_with_noise_with_transform(self):
        self._test_selection(add_noise=True, device='cuda',
                             apply_transform=True)

    def _test_selection(self, add_noise: bool = False, apply_transform: bool = False, device: Optional[str] = None):
        print('\nTest selection of features in a linear transformation setting')
        d: int = np.random.randint(low=10, high=20)
        n: int = np.random.randint(low=10000, high=20000)
        n_features: int = d // 3
        x: np.ndarray = np.random.uniform(size=(n, d))
        features = list(np.random.choice(d, replace=False, size=n_features))
        y: np.array = x[:, features]
        if apply_transform:
            tt = np.expand_dims(
                special_ortho_group.rvs(n_features),
                axis=0
            )
            yy = np.expand_dims(y, axis=2)
            y = (tt @ yy)[:, :, 0]
        if add_noise:
            y += .1 * np.std(y) * np.random.uniform(size=y.shape)
        if use_pyhsiclasso:
            print('Using pyHSICLasso for reconciliation purposes')
            pyhsiclasso_selection = pyhsiclasso(
                x, y, n_features, 500)
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

        selector = Selector(x, y)
        selection = selector.select(
            n_features, batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, device=device)
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

        # Test autoselection - We do not provide the number of features that should be selected
        autoselection = selector.autoselect(
            batch_size=len(x) // 4, minibatch_size=400,  number_of_epochs=3, threshold=5e-3, device=device)
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


if __name__ == '__main__':
    unittest.main()
