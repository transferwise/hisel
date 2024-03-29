from dataclasses import make_dataclass
import timeit
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from hisel import kernels
from hisel.kernels import Device
from hisel import cudakernels


def hisel_compute_gram_matrix(x, batch_size):
    rbf_kernel = kernels.KernelType.RBF
    l = 1.
    gram = kernels.apply_feature_map(
        rbf_kernel,
        x,
        l,
        batch_size,
        is_multivariate=False,
        device=Device.CPU,
    )
    return gram


def cudahisel_compute_gram_matrix(x, batch_size):
    rbf_kernel = kernels.KernelType.RBF
    l = 1.
    gram = cudakernels.apply_feature_map(
        rbf_kernel,
        x,
        l,
        batch_size,
        is_multivariate=False,
        device=Device.GPU,
    )
    return gram


class PyHSICLasso:
    # This class is used to reproduce the code of the package `pyHSICLasso` that we need to compare with `hisel`

    @staticmethod
    def compute_gram_matrix(x, batch_size, parallel: bool = False):
        # This function faithfully reproduces the inner working
        # of the function pyHSICLasso.hsic_lasso.hsic_lasso
        # responsible for computing the Gram matrix.
        # See
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/hsic_lasso.py#L18

        d, n = x.shape
        discarded = 0
        if parallel:
            result = Parallel(n_jobs=-1)([
                delayed(PyHSICLasso.parallel_compute_kernel)(
                    np.reshape(x[k, :], (1, n)),
                    'Gaussian',
                    k,
                    batch_size,
                    1,
                    n,
                    discarded)
                for k in tqdm(range(d))
            ])
        else:
            result = []
            for k in tqdm(range(d)):
                X = PyHSICLasso.parallel_compute_kernel(
                    x[[k], :], 'Gaussian', k, batch_size, 1, n, discarded)
                result.append(X)
        result = dict(result)
        K = np.array([result[k] for k in range(d)]).T
        return K

    @staticmethod
    def compute_kernel(x, kernel, B=0, M=1, discarded=0):
        # This method reproduces the following
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/hsic_lasso.py#L53

        d, n = x.shape

        H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
        K = np.zeros(n * B * M, dtype=np.float32)

        # Normalize data
        if kernel == "Gaussian":
            x = (x / (x.std() + 10e-20)).astype(np.float32)

        st = 0
        ed = B ** 2
        index = np.arange(n)
        for m in range(M):
            np.random.seed(m)
            index = np.random.permutation(index)

            for i in range(0, n - discarded, B):
                j = min(n, i + B)

                if kernel == 'Gaussian':
                    k = PyHSICLasso.kernel_gaussian(
                        x[:, index[i:j]], x[:, index[i:j]], np.sqrt(d))
                elif kernel == 'Delta':
                    k = PyHSICLasso.kernel_delta_norm(
                        x[:, index[i:j]], x[:, index[i:j]])

                k = np.dot(np.dot(H, k), H)

                # Normalize HSIC tr(k*k) = 1
                k = k / (np.linalg.norm(k, 'fro') + 10e-10)
                K[st:ed] = k.flatten()
                st += B ** 2
                ed += B ** 2

        return K

    @staticmethod
    def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded):
        # This method reproduces the following:
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/hsic_lasso.py#L89

        return (feature_idx, PyHSICLasso.compute_kernel(x, kernel, B, M, discarded))

    @staticmethod
    def kernel_delta_norm(X_in_1, X_in_2):
        # This method reproduces the following:
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/kernel_tools.py#L13
        n_1 = X_in_1.shape[1]
        n_2 = X_in_2.shape[1]
        K = np.zeros((n_1, n_2))
        u_list = np.unique(X_in_1)
        for ind in u_list:
            c_1 = np.sqrt(np.sum(X_in_1 == ind))
            c_2 = np.sqrt(np.sum(X_in_2 == ind))
            ind_1 = np.where(X_in_1 == ind)[1]
            ind_2 = np.where(X_in_2 == ind)[1]
            K[np.ix_(ind_1, ind_2)] = 1 / c_1 / c_2
        return K

    @staticmethod
    def kernel_delta(X_in_1, X_in_2):
        # This method reproduces the following:
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/kernel_tools.py#L27
        n_1 = X_in_1.shape[1]
        n_2 = X_in_2.shape[1]
        K = np.zeros((n_1, n_2))
        u_list = np.unique(X_in_1)
        for ind in u_list:
            ind_1 = np.where(X_in_1 == ind)[1]
            ind_2 = np.where(X_in_2 == ind)[1]
            K[np.ix_(ind_1, ind_2)] = 1
        return K

    @staticmethod
    def kernel_gaussian(X_in_1, X_in_2, sigma):
        # This method reproduces the following:
        # https://github.com/riken-aip/pyHSICLasso/blob/400afe9347bf0ed58d97b9e39b911b44c45bebff/pyHSICLasso/kernel_tools.py#L39
        n_1 = X_in_1.shape[1]
        n_2 = X_in_2.shape[1]
        X_in_12 = np.sum(np.power(X_in_1, 2), 0)
        X_in_22 = np.sum(np.power(X_in_2, 2), 0)
        dist_2 = np.tile(X_in_22, (n_1, 1)) + \
            np.tile(X_in_12, (n_2, 1)).transpose() - \
            2 * np.dot(X_in_1.T, X_in_2)
        K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
        return K


class Experiment:
    def __init__(self,
                 num_samples=5000,
                 num_features=200,
                 batch_size=1000,
                 ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.batch_size = batch_size
        self.x = np.random.uniform(low=0., high=1.,
                                   size=(num_features, num_samples))

    def run_hisel(self):
        return hisel_compute_gram_matrix(self.x, self.batch_size)

    def run_cudahisel(self):
        return cudahisel_compute_gram_matrix(self.x, self.batch_size)

    def run_pyhsiclasso(self):
        return PyHSICLasso.compute_gram_matrix(
            self.x, self.batch_size, False)


def test_num_samples():
    num_features = 300
    batch_size = 800
    num_samples = 1600 * np.arange(2, 8, dtype=int)
    num_runs = 5
    data = []
    Result = make_dataclass("Result",
                            [
                                ("num_samples", int),
                                ("num_features", int),
                                ("batch_size", int),
                                ("hisel_cpu_time", float),
                                ("hisel_gpu_time", float),
                                ("pyhsiclasso_time", float),
                            ])
    for n in num_samples:
        experiment = Experiment(
            n, num_features, batch_size)
        hisel_cpu_time = timeit.timeit(
            'experiment.run_hisel()',
            globals=locals(),
            number=num_runs)
        hisel_cpu_time /= num_runs
        hisel_gpu_time = timeit.timeit(
            'experiment.run_cudahisel()',
            globals=locals(),
            number=num_runs)
        hisel_gpu_time /= num_runs
        pyhsiclasso_time = timeit.timeit(
            'experiment.run_pyhsiclasso()',
            globals=locals(),
            number=num_runs)
        pyhsiclasso_time /= num_runs
        del experiment
        result = Result(
            n,
            num_features,
            batch_size,
            hisel_cpu_time,
            hisel_gpu_time,
            pyhsiclasso_time
        )
        print(result)
        data.append(result)
    df = pd.DataFrame(data)
    df.to_csv("gram_runtimes_by_num_samples.csv", index=False)
    print(df)


def main():

    experiment = Experiment()

    # Compute Gram matrix using hisel
    hisel_time = timeit.timeit(
        'experiment.run_hisel()',
        globals=locals(),
        number=3)
    print('\n#################################################################')
    print(f'# hisel_time: {round(hisel_time, 6)}')
    print('#################################################################\n\n')

    # Compute Gram matrix using hisel
    cudahisel_time = timeit.timeit(
        'experiment.run_cudahisel()',
        globals=locals(),
        number=3)
    print('\n#################################################################')
    print(f'# cudahisel_time: {round(cudahisel_time, 6)}')
    print('#################################################################\n\n')

    # Compute Gram matrix using pyHSICLasso
    pyhsiclasso_time = timeit.timeit(
        'experiment.run_pyhsiclasso()',
        globals=locals(),
        number=3)
    print('\n#################################################################')
    print(f'# pyhsiclasso_time: {round(pyhsiclasso_time, 6)}')
    print('#################################################################\n\n\n')


if __name__ == '__main__':
    main()
