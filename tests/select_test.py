import unittest
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import HSICSelector as Selector, FeatureType
from hisel.kernels import Device
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

USE_PYHSICLASSO = True
try:
    import pyHSICLasso
except (ModuleNotFoundError, ImportError):
    USE_PYHSICLASSO = False

QUICK_TEST = True
SKIP_CUDA = False
USE_PYHSICLASSO = False if QUICK_TEST else USE_PYHSICLASSO
SKLEARN_RECON = True


def pyhsiclasso(x, y, xfeattype, yfeattype, n_features: int, minibatch_size=500):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    discrete_x = False  # xfeattype == FeatureType.DISCR
    if yfeattype == FeatureType.DISCR:
        lasso.classification(n_features,
                             B=minibatch_size,
                             discrete_x=discrete_x)
    else:
        lasso.regression(n_features,
                         B=minibatch_size,
                         discrete_x=discrete_x)
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

    def test_regression_with_non_linear_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(
            xfeattype, yfeattype,
            add_noise=False,
            apply_transform=False,
            apply_non_linear_transform=True)

    def test_regression_with_non_linear_transform_and_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(
            xfeattype, yfeattype,
            add_noise=True,
            apply_transform=False,
            apply_non_linear_transform=True)

    def test_regression_with_linear_and_non_linear_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(
            xfeattype, yfeattype,
            add_noise=False,
            apply_transform=True,
            apply_non_linear_transform=True)

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

    def test_mixed_feature_regression_no_noise(self):
        xfeattype = FeatureType.BOTH
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype, add_noise=False)

    def test_mixed_feature_regression_with_transform(self):
        xfeattype = FeatureType.BOTH
        yfeattype = FeatureType.CONT
        self._test_selection(
            xfeattype, yfeattype,
            apply_transform=True, add_noise=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, device=Device.GPU)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=True, device=Device.GPU)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_no_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype,
                             add_noise=False, device=Device.GPU, apply_transform=True)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_regression_with_noise_with_transform(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.CONT
        self._test_selection(xfeattype, yfeattype, add_noise=True, device=Device.GPU,
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
                             add_noise=False, device=Device.GPU)

    @unittest.skipIf(SKIP_CUDA, 'cuda not available')
    def test_cuda_classification_with_noise(self):
        xfeattype = FeatureType.CONT
        yfeattype = FeatureType.DISCR
        self._test_selection(xfeattype, yfeattype,
                             add_noise=True, device=Device.GPU)

    def _test_selection(
        self,
        xfeattype: FeatureType,
        yfeattype: FeatureType,
        add_noise: bool = False,
        apply_transform: bool = False,
        device: Device = Device.CPU,
        apply_non_linear_transform: bool = False,
    ):
        print('\n\n\n##############################################################################')
        print('Test selection of features in a (non-)linear  transformation setting')
        print(
            '##############################################################################')
        print(f'Feature type of x: {xfeattype}')
        print(f'Feature type of y: {yfeattype}')
        print(f'Apply linear transform: {apply_transform}')
        print(f'Apply non-linear transform: {apply_non_linear_transform}')
        print(f'Noisy target: {add_noise}')
        print(f'device: {device}')
        d: int = np.random.randint(low=15, high=25)
        minibatch_size: int = np.random.randint(low=500, high=1000)
        n: int = minibatch_size * \
            (np.random.randint(low=5000, high=10000) // minibatch_size)
        batch_size: int = n
        n_features: int = d // 3
        features = list(np.random.choice(d, replace=False, size=n_features))
        x: np.ndarray
        y: np.ndarray
        if xfeattype == FeatureType.DISCR:
            ms = np.random.randint(low=2, high=2*n_features, size=(d,))
            xs = [np.random.randint(m, size=(n, 1)) for m in ms]
            x = np.concatenate(xs, axis=1)
            split = None
        elif xfeattype == FeatureType.BOTH:
            split: int = np.random.randint(low=3, high=d-1)
            xcat = np.random.randint(10, size=(n, split))
            xcont = np.random.uniform(size=(n, d-split))
            x = np.concatenate((xcat, xcont), axis=1)
        else:
            x = np.random.uniform(size=(n, d))
            split = None
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
            if apply_non_linear_transform:
                u = np.sum(u, axis=1, keepdims=True)
                u /= np.max(np.abs(u), axis=None)
                y = np.sin(4 * np.pi * u)
            else:
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
                x, y, xfeattype, yfeattype, n_features, minibatch_size)
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
            xfeattype=xfeattype,
            yfeattype=yfeattype,
            catcont_split=split,
        )
        num_to_select = n_features
        selected_features = selector.select(
            num_to_select,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            number_of_epochs=3,
            device=device)
        selection = [int(feat.split('f')[-1])
                     for feat in selected_features]
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
            if not QUICK_TEST:
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
        if QUICK_TEST:
            if (xfeattype == FeatureType.DISCR or xfeattype == FeatureType.BOTH or yfeattype == FeatureType.DISCR):
                grace = 13 if xfeattype == FeatureType.BOTH else 9
            elif apply_non_linear_transform and apply_transform:
                grace = 11
            elif apply_non_linear_transform:
                grace = 5
            elif 6 * y.shape[1] < n_features:
                grace = 5
            else:
                grace = 3
            diff_left = set(selection).difference(set(features))
            diff_right = set(features).difference(set(selection))
            diff = diff_left.union(diff_right)
            self.assertLess(
                len(diff),
                grace,
                msg=(f'\nExpected features: {sorted(features)}\n'
                     f'Selected features: {sorted(selection)}'
                     )
            )
        else:
            self.assertEqual(
                set(selection),
                set(features),
                msg=(f'\nExpected features: {sorted(features)}\n'
                     f'Selected features: {sorted(selection)}'
                     )
            )

        if QUICK_TEST:
            return
        # Test autoselection - We do not provide the number of features that should be selected
        autoselected_features = selector.autoselect(
            batch_size=len(x) // 4,
            minibatch_size=400,
            number_of_epochs=3,
            threshold=3e-2,
            device=device)
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


class BothKernelTest(unittest.TestCase):

    def test_classification(self):
        xfeattype = FeatureType.BOTH
        yfeattype = FeatureType.DISCR
        d = 20
        n = 5000
        for split in (0, 2, 10, 18, 20):
            self._test(
                xfeattype,
                yfeattype,
                d,
                split,
                n
            )

    def test_regression(self):
        xfeattype = FeatureType.BOTH
        yfeattype = FeatureType.CONT
        d = 20
        n = 5000
        split = 10
        for split in (0, 2, 10, 18, 20):
            self._test(
                xfeattype,
                yfeattype,
                d,
                split,
                n
            )

    def _test(
        self,
            xfeattype: FeatureType,
            yfeattype: FeatureType,
            d: int,
            split: int,
            n: int):
        print('\n\n\n##############################################################################')
        print('Test single selection from mixed features')
        print(
            '##############################################################################')
        print(f'Feature type of x: {xfeattype}')
        print(f'Feature type of y: {yfeattype}')
        print(f'Total number of features: {d}')
        print(f'Split: {split}')
        relevant_features = np.random.choice(d, size=(1, ), replace=False)
        xdiscr = np.random.randint(10, size=(n, split))
        xcont = np.random.uniform(low=-10., high=10, size=(n, d-split))
        x = np.concatenate((xdiscr, xcont), axis=1)
        if yfeattype == FeatureType.CONT:
            t = x[:, relevant_features] / \
                np.max(np.abs(x[:, relevant_features]), axis=None)
            y = 10 * np.sin(4 * np.pi * t)
        elif yfeattype == FeatureType.DISCR:
            y = np.random.randint(
                10) + np.abs(x[:, relevant_features]).astype(int)
        else:
            raise ValueError(yfeattype)
        selector = Selector(
            x, y,
            xfeattype=xfeattype,
            yfeattype=yfeattype,
            catcont_split=split,
        )
        num_to_select = len(relevant_features)
        selected_features = selector.select(
            num_to_select, batch_size=n, minibatch_size=800, number_of_epochs=3)
        selection = [int(feat.split('f')[-1])
                     for feat in selected_features]
        print(f'Expected features:\n{sorted(relevant_features)}')
        print(
            f'hisel selected features:\n{sorted(selection)}')
        self.assertEqual(
            len(selection),
            len(relevant_features),
        )
        self.assertEqual(
            set(selection),
            set(relevant_features)
        )


if __name__ == '__main__':
    unittest.main()
