import timeit
import numpy as np
import cupy as cp
from hisel import lar
from pyHSICLasso import nlars


class Experiment:
    def __init__(self, n, d, a):
        x = np.random.uniform(size=(n, d))
        beta = np.random.permutation(np.vstack(
            [
                np.random.uniform(low=0., high=1., size=(a, 1)),
                np.zeros((d-a, 1), dtype=np.float32)
            ]
        ))
        y = x @ beta + np.random.uniform(low=-1e-2, high=1e-2, size=(n, 1))

        self.a = a
        self.x = x
        self.x_ = cp.array(x)
        self.y = y
        self.y_ = cp.array(y)
        self.beta = beta

    def run_hisel(self):
        feats, _ = lar.solve(self.x, self.y, self.a)

    def run_hisel_from_cupy_arrays(self):
        x = cp.asnumpy(self.x_)
        y = cp.asnumpy(self.y_)
        feats, _ = lar.solve(x, y, self.a)

    def run_pyhsiclasso(self):
        _ = nlars.nlars(self.x, self.x.T @ self.y, self.a, 3)


def main():
    n = 100000
    d = 500
    a = 100
    experiment = Experiment(n, d, a)
    hisel_time = timeit.timeit(
        'experiment.run_hisel()',
        globals=locals(),
        number=1
    )
    print('\n#################################################################')
    print(f'# hisel_time: {round(hisel_time, 6)}')
    print('#################################################################\n\n')

    hisel_gpu_time = timeit.timeit(
        'experiment.run_hisel_from_cupy_arrays()',
        globals=locals(),
        number=1
    )
    print('\n#################################################################')
    print(f'# hisel_gpu_time: {round(hisel_gpu_time, 6)}')
    print('#################################################################\n\n')

    pyhsiclasso_time = timeit.timeit(
        'experiment.run_pyhsiclasso()',
        globals=locals(),
        number=1
    )

    print('\n#################################################################')
    print(f'# pyhsiclasso_time: {round(pyhsiclasso_time, 6)}')
    print('#################################################################\n\n')


if __name__ == '__main__':
    main()
