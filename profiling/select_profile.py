import pstats
from pstats import SortKey
import cProfile
from tests.select_test import SelectorTest
import numpy as np
from scipy.stats import special_ortho_group

from hisel.select import HSICSelector as Selector, FeatureType
from hisel.kernels import Device


class SelectProfiler:
    def __init__(
        self,
        xfeattype: FeatureType,
        yfeattype: FeatureType,
        add_noise: bool = False,
        apply_transform: bool = False,
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
        d: int = 200
        minibatch_size: int = 500
        n: int = 4000
        batch_size: int = n
        n_features: int = 30
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

        self.x = x
        self.y = y
        self.xfeattype = xfeattype
        self.yfeattype = yfeattype
        self.split = split
        self.features = features
        self.n_features = n_features
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

    def run_on_cpu(self):
        self._run(Device.CPU)

    def run_on_gpu(self):
        self._run(Device.GPU)

    def _run(self, device: Device = Device.CPU):

        selector = Selector(
            self.x, self.y,
            xfeattype=self.xfeattype,
            yfeattype=self.yfeattype,
            catcont_split=self.split,
        )
        num_to_select = self.n_features
        selection = selector.select(
            num_to_select,
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            number_of_epochs=3,
            device=device,
            return_index=True)
        print(f'Expected features:\n{sorted(self.features)}')
        print(
            f'hisel selected features:\n{sorted(selection)}')
        del selector


def main():
    profiler = SelectProfiler(
        xfeattype=FeatureType.CONT,
        yfeattype=FeatureType.CONT,
        add_noise=True,
        apply_transform=False,
    )
    cProfile.runctx('profiler.run_on_cpu()', globals=globals(),
                    locals=locals(), filename='cpu_select_profile')
    p = pstats.Stats('cpu_select_profile')
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        30)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/select.py:', 20)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/kernels.py:', 20)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(
        'lar.py:', 20)

    cProfile.runctx('profiler.run_on_gpu()',  globals=globals(),
                    locals=locals(), filename='gpu_select_profile')
    p_gpu = pstats.Stats('gpu_select_profile')
    p_gpu.sort_stats(SortKey.CUMULATIVE).print_stats(
        30)
    p_gpu.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/select.py:', 20)
    p_gpu.sort_stats(SortKey.CUMULATIVE).print_stats(
        'hisel/cudakernels.py:', 20)
    p_gpu.sort_stats(SortKey.CUMULATIVE).print_stats(
        'lar.py:', 20)


if __name__ == '__main__':
    main()
