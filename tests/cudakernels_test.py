import unittest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hisel import kernels, cudakernels
from hisel.kernels import Device
import datetime

CUPY_AVAILABLE = True
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    print('Could not import cupy!')
    cp = np
    CUPY_AVAILABLE = False


class CudaKernelTest(unittest.TestCase):

    def test_rbf(self):
        print('\n...Test RBF...')
        kernel_type = kernels.KernelType.RBF
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        rbf = RBF(l)
        x = np.random.uniform(size=(d, n))
        x /= np.std(x, axis=1, keepdims=True) + 1e-19
        k = np.zeros((d, n, n))
        g = np.zeros((d, n, n))
        if CUPY_AVAILABLE:
            x_ = cp.array(x)
            g_ = cp.array(g)
        else:
            x_ = x
            g_ = g
        for i in range(d):
            k[i, :, :] = rbf(x[[i], :].T)
            g_[i, :, :] = cudakernels.multivariate(
                x_[[i], :], l, kernel_type)

        f_ = cudakernels.featwise(x_, l, kernel_type)
        ff = kernels.featwise(x, l, kernel_type)
        if CUPY_AVAILABLE:
            f = cp.asnumpy(f_)
            g = cp.asnumpy(g_)
        else:
            f = np.array(f_, copy=True)
            g = np.array(g_, copy=True)
        self.assertTrue(
            np.allclose(
                f, k
            )
        )
        self.assertTrue(
            np.allclose(
                g, k
            )
        )
        self.assertTrue(
            np.allclose(
                f, ff
            ))

    def test_delta(self):
        print('\n...Test DELTA...')
        kernel_type = kernels.KernelType.DELTA
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l = 1.
        m: int = np.random.randint(low=6, high=12)
        x = np.random.randint(m, size=(d, n))
        g = np.zeros((d, n, n))
        if CUPY_AVAILABLE:
            x_ = cp.array(x)
            g_ = cp.array(g)
        else:
            x_ = x
            g_ = g
        for i in range(d):
            g_[i, :, :] = cudakernels.multivariate(
                x_[[i], :], l, kernel_type)

        f_ = cudakernels.featwise(x_, l, kernel_type)
        ff = kernels.featwise(x, l, kernel_type)
        if CUPY_AVAILABLE:
            f = cp.asnumpy(f_)
            g = cp.asnumpy(g_)
        else:
            f = np.array(f_, copy=True)
            g = np.array(g_, copy=True)

        self.assertTrue(
            np.allclose(
                g, f
            )
        )
        self.assertTrue(
            np.allclose(
                f, ff
            ))

    def test_both(self):
        print('\n...Test BOTH...')
        kernel_type = kernels.KernelType.BOTH
        d: int = np.random.randint(low=6, high=15)
        split: int = np.random.randint(low=2, high=d-1)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        m: int = np.random.randint(low=6, high=12)
#         print(f'd: {d}')
#         print(f'split: {split}')
#         print(f'n: {n}')
        rbf = RBF(l)
        xcat = np.random.randint(m, size=(split, n))
        xcont = np.random.uniform(size=(d-split, n))
        xcont /= np.std(xcont, axis=1, keepdims=True) + 1e-19
        x = np.concatenate((xcat, xcont), axis=0)
        k = np.zeros((d, n, n))
        g = np.zeros((d, n, n))
        if CUPY_AVAILABLE:
            x_ = cp.array(x)
            g_ = cp.array(g)
        else:
            x_ = x
            g_ = g
        for i in range(split):
            g_[i, :, :] = cudakernels.multivariate(
                x_[[i], :].astype(int),
                1.,
                kernels.KernelType.DELTA)
        for i in range(split, d):
            k[i, :, :] = rbf(x[[i], :].T)
            g_[i, :, :] = kernels.multivariate(
                x_[[i], :], l, kernels.KernelType.RBF)

        f_ = cudakernels.featwise(x_, l, kernel_type, split)
        ff = kernels.featwise(x, l, kernel_type, split)
        if CUPY_AVAILABLE:
            f = cp.asnumpy(f_)
            g = cp.asnumpy(g_)
        else:
            f = np.array(f_, copy=True)
            g = np.array(g_, copy=True)

        self.assertTrue(
            np.allclose(
                f[split:], k[split:]
            )
        )
        self.assertTrue(
            np.allclose(
                g, f
            )
        )
        self.assertTrue(
            np.allclose(
                ff, f
            )
        )

    def test_apply_rbf_feature_map(self):
        kernel_type = kernels.KernelType.RBF
        self._test_apply_feature_map(kernel_type)

    def test_apply_delta_feature_map(self):
        kernel_type = kernels.KernelType.DELTA
        self._test_apply_feature_map(kernel_type)

    def _test_apply_feature_map(self, kernel_type):
        print('\n...Test apply_feature_map...')
        print(f'kernel_type: {kernel_type}')
        d: int = np.random.randint(low=5, high=12)
        n: int = np.random.randint(low=30000, high=35000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        gram_dim: int = num_batches * batch_size**2
        if kernel_type == kernels.KernelType.DELTA:
            x = np.random.randint(10, size=(d, n))
        else:
            x = np.random.uniform(size=(d, n))
        if CUPY_AVAILABLE:
            x_ = cp.array(x)
        else:
            x_ = x

        # Execution on CPU
        t0 = datetime.datetime.now()
        phi_cpu = kernels.apply_feature_map(
            kernel_type, x, l, batch_size, device=Device.CPU
        )
        t1 = datetime.datetime.now()
        dt_cpu = t1 - t0
        cpu_runtime = dt_cpu.seconds + 1e-6 * dt_cpu.microseconds

        self.assertEqual(
            phi_cpu.shape,
            (gram_dim, d)
        )
        print(f'runtime on cpu: {cpu_runtime} seconds')

        # Execution on GPU
        t0 = datetime.datetime.now()
        phi: np.ndarray = cudakernels.apply_feature_map(
            kernel_type, x_, l, batch_size, device=Device.GPU
        )
        t1 = datetime.datetime.now()
        dt_gpu = t1 - t0
        gpu_runtime = dt_gpu.seconds + 1e-6 * dt_gpu.microseconds
        self.assertEqual(
            phi.shape,
            (gram_dim, d)
        )
        print(f'runtime with gpu execution: {gpu_runtime} seconds')

        # check that cpu and gpu execution yield the same results
        self.assertTrue(
            np.allclose(
                phi,
                phi_cpu
            )
        )


if __name__ == '__main__':
    unittest.main()
