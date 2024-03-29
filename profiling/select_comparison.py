import timeit
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import HSICSelector as Selector, FeatureType
from hisel.kernels import Device

import pyHSICLasso


def pyhsiclasso(x, y, xfeattype, yfeattype,
                n_features: int, batch_size: int = 500, number_of_epochs: int = 3):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    discrete_x = xfeattype == FeatureType.DISCR
    if yfeattype == FeatureType.DISCR:
        lasso.classification(n_features, B=batch_size,
                             discrete_x=discrete_x, M=number_of_epochs)
    else:
        lasso.regression(n_features, B=batch_size,
                         discrete_x=discrete_x, M=number_of_epochs)

    selection = np.array(lasso.A, copy=True)
    del lasso
    return selection


class Experiment:

    def __init__(
        self,
        xfeattype: FeatureType,
        yfeattype: FeatureType,
        add_noise: bool = False,
        apply_transform: bool = False,
        batch_size: int = 500,
        number_of_epochs: int = 3,
    ):
        print('\n\n\n##############################################################')
        print('# Test selection of features in a linear transformation setting')
        print('##############################################################')
        print(f'Feature type of x: {xfeattype}')
        print(f'Feature type of y: {yfeattype}')
        print(f'Apply transform: {apply_transform}')
        print(f'Noisy target: {add_noise}')
        print(f'Number of epochs: {number_of_epochs}')
        print(f'Batch size: {batch_size}')

        d: int = np.random.randint(low=300, high=400)
        n: int = batch_size * \
            (np.random.randint(low=10000, high=12000) // batch_size)
        n_features: int = d // np.random.randint(low=15, high=20)
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
            y = np.sum(u, axis=1, keepdims=True)
        elif yfeattype == FeatureType.DISCR:
            y = np.zeros(shape=(n, 1), dtype=int)
            for i in range(1, n_features):
                y += np.asarray(u[:, [i-1]] > u[:, [i]], dtype=int)
        else:
            raise ValueError(yfeattype)

        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.d = d
        self.n = n
        self.n_features = n_features
        self.features = features
        self.xfeattype = xfeattype
        self.yfeattype = yfeattype
        self.x = x
        self.y = y

    def run_pyhsiclasso(self):
        pyhsiclasso_selection = pyhsiclasso(
            self.x, self.y, self.xfeattype, self.yfeattype,
            self.n_features, self.batch_size, self.number_of_epochs)
        print(
            f'pyHSICLasso selected features:\n{sorted(pyhsiclasso_selection)}')

        if not set(pyhsiclasso_selection) == set(self.features):
            msg = (
                f'\npyhsiclasso_selection: {sorted(pyhsiclasso_selection)}'
                f'\nfeatures: {sorted(self.features)}\n\n'
            )
            print(
                f'WARNING: pyHSICLasso did not perform an exact selection:\n{msg}')

        return pyhsiclasso_selection

    def run_hisel_on_cpu(self):
        self._run_hisel(Device.CPU)

    def run_hisel_on_gpu(self):
        self._run_hisel(Device.GPU)

    def _run_hisel(self, device: Device = Device.CPU):
        selector = Selector(
            self.x, self.y,
            xfeattype=self.xfeattype,
            yfeattype=self.yfeattype
        )
        selection = selector.select(
            self.n_features,
            batch_size=len(self.x),
            minibatch_size=self.batch_size,
            number_of_epochs=self.number_of_epochs,
            device=device,
            return_index=True,
        )
        print(
            f'hisel selected features:\n{sorted(selection)}')

        if not set(selection) == set(self.features):
            msg = (f'Expected features: {sorted(self.features)}\n'
                   f'Selected features: {sorted(selection)}'
                   )
            print(
                f'WARNING: hisel did not perform an exact selection:\n{msg}')

        del selector
        return selection


def test_regression_with_noise():
    xfeattype = FeatureType.CONT
    yfeattype = FeatureType.CONT
    batch_size = 750
    number_of_epochs = 1
    return Experiment(xfeattype, yfeattype,
                      add_noise=True,
                      apply_transform=False,
                      batch_size=batch_size,
                      number_of_epochs=number_of_epochs)


def test_regression_with_noise_with_transform():
    xfeattype = FeatureType.CONT
    yfeattype = FeatureType.CONT
    batch_size = 750
    number_of_epochs = 1
    return Experiment(xfeattype, yfeattype,
                      add_noise=True,
                      apply_transform=True,
                      batch_size=batch_size,
                      number_of_epochs=number_of_epochs)


def main():
    regression_experiment = test_regression_with_noise()

    pyhsiclasso_time = timeit.timeit(
        'regression_experiment.run_pyhsiclasso()',
        number=1,
        globals=locals(),
    )
    print('\n#################################################################')
    print(f'# pyhsiclasso_time: {round(pyhsiclasso_time, 6)}')
    print('#################################################################\n\n\n')

    hisel_cpu_time = timeit.timeit(
        'regression_experiment.run_hisel_on_cpu()',
        number=1,
        globals=locals(),
    )
    print('\n#################################################################')
    print(f'# hisel_cpu_time: {round(hisel_cpu_time, 6)}')
    print('#################################################################\n\n')

    hisel_gpu_time = timeit.timeit(
        'regression_experiment.run_hisel_on_gpu()',
        number=1,
        globals=locals(),
    )
    print('\n#################################################################')
    print(f'# hisel_gpu_time: {round(hisel_gpu_time, 6)}')
    print('#################################################################\n\n')


if __name__ == '__main__':
    main()
