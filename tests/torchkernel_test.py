import datetime
import unittest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hisel import kernels
SKIP = False
SKIP_CUDA = True
try:
    import torch
    from torch import Tensor
    from hisel import torchkernels
    SKIP_CUDA = not torch.cuda.is_available()
except (ModuleNotFoundError, ImportError):
    SKIP = True


class KernelTest(unittest.TestCase):
    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_rbf_featwise(self):
        kernel_type = torchkernels.KernelType.RBF
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)

        x = np.random.uniform(size=(d, n))
        x_torch = torch.from_numpy(x)

        f = kernels.featwise(x, l, kernel_type)
        f_torch = torchkernels.featwise(
            x_torch, l, kernel_type).detach().cpu().numpy()

        self.assertEqual(f.shape, f_torch.shape)

        self.assertTrue(
            np.allclose(
                f, f_torch,
            )
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_delta_featwise(self):
        kernel_type = torchkernels.KernelType.DELTA
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        m: int = np.random.randint(low=6, high=12)

        x = np.random.randint(m, size=(d, n))
        x_torch = torch.from_numpy(x)

        f = kernels.featwise(x, l, kernel_type)
        f_torch = torchkernels.featwise(
            x_torch, l, kernel_type).detach().cpu().numpy()

        self.assertEqual(f.shape, f_torch.shape)

        self.assertTrue(
            np.allclose(
                f, f_torch,
            )
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_rbf(self):
        kernel_type = torchkernels.KernelType.RBF
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        rbf = RBF(l)
        x = torch.randn(d, n)
        k = np.zeros((d, n, n))
        g = torch.zeros((d, n, n))
        for i in range(d):
            k[i, :, :] = rbf(x[[i], :].T)
            g[i, :, :] = torchkernels.multivariate(
                x[[i], :], x[[i], :], l, kernel_type)

        f = torchkernels.featwise(x, l, kernel_type)

        g = g.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

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

    @unittest.skipIf(SKIP, 'torch not available')
    def test_delta(self):
        kernel_type = torchkernels.KernelType.DELTA
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        m: int = np.random.randint(low=6, high=12)
        x = torch.randint(m, size=(d, n))
        g = torch.zeros((d, n, n))
        for i in range(d):
            g[i, :, :] = torchkernels.multivariate(
                x[[i], :], x[[i], :], l, kernel_type)

        f = torchkernels.featwise(x, l, kernel_type)

        g = g.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        self.assertTrue(
            np.allclose(
                f, g
            )
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_multivariate_phi_rbf(self):
        kernel_type = torchkernels.KernelType.RBF
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)

        x = np.random.uniform(size=(d, n))
        x_torch = torch.from_numpy(x)

        g = kernels.multivariate_phi(x, l, kernel_type)
        _g_torch = torchkernels.multivariate_phi(x_torch, l, kernel_type)
        g_torch = _g_torch.detach().cpu().numpy()
        self.assertEqual(
            g.shape,
            g_torch.shape
        )
        self.assertTrue(
            np.allclose(
                g, g_torch
            ))

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_multivariate_phi_delta(self):
        kernel_type = torchkernels.KernelType.DELTA
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        m: int = np.random.randint(low=6, high=18)

        x = np.random.randint(m, size=(d, n))
        x_torch = torch.from_numpy(x)

        g = kernels.multivariate_phi(x, l, kernel_type)
        _g_torch = torchkernels.multivariate_phi(x_torch, l, kernel_type)
        g_torch = _g_torch.detach().cpu().numpy()
        self.assertEqual(
            g.shape,
            g_torch.shape
        )
        self.assertTrue(
            np.allclose(
                g, g_torch
            ))

    @unittest.skipIf(SKIP, 'torch not available')
    def test_centering_matrix(self):
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        h: Tensor = torchkernels._centering_matrix(d, n)
        self.assertEqual(
            h.size(),
            (d, n, n)
        )
        h_: Tensor = torch.eye(n, dtype=torch.float64) - \
            torch.ones(n, dtype=torch.float64) / n
        self.assertTrue(
            torch.allclose(
                h_, h[0, :, :]
            )
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_centering_matrix(self):
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        h: np.ndarray = kernels._centering_matrix(d, n)
        h_torch: np.ndarray = torchkernels._centering_matrix(
            d, n).detach().cpu().numpy()
        self.assertEqual(
            h.shape,
            h_torch.shape
        )
        self.assertTrue(
            np.allclose(h, h_torch)
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_make_batches(self):
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=10000, high=20000)
        num_batches = 10
        batch_size = n // num_batches
        x = np.random.uniform(size=(d, n))
        x_torch = torch.from_numpy(x)
        assert x.shape == x_torch.size()
        batches = kernels._make_batches(x, batch_size)
        batches_torch = torchkernels._make_batches(x_torch, batch_size)
        self.assertEqual(
            len(batches), len(batches_torch)
        )
        for b, bt in zip(batches, batches_torch):
            self.assertEqual(
                b.shape,
                bt.size()
            )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_apply_rbf_feature_map(self):
        kernel_type = torchkernels.KernelType.RBF
        self._test_torch_v_numpy_apply_feature_map(kernel_type)

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_apply_delta_feature_map(self):
        kernel_type = torchkernels.KernelType.DELTA
        self._test_torch_v_numpy_apply_feature_map(kernel_type)

    def _test_torch_v_numpy_apply_feature_map(self, kernel_type):
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=10000, high=20000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        if kernel_type == torchkernels.KernelType.DELTA:
            m = np.random.randint(low=6, high=20)
            x = np.random.randint(m, size=(d, n))
        else:
            x = np.random.uniform(size=(d, n))
        x_torch = torch.from_numpy(x)
        assert x.shape == x_torch.size()
        phi: np.ndarray = kernels.apply_feature_map(
            kernel_type, x, l, batch_size
        )
        phi_torch: np.ndarray = torchkernels.apply_feature_map(
            kernel_type, x_torch, l, batch_size
        )
        gram_dim: int = num_batches * batch_size**2
        self.assertEqual(
            phi.shape,
            (gram_dim, d)
        )
        self.assertEqual(
            phi_torch.shape,
            (gram_dim, d)
        )
        self.assertTrue(
            np.allclose(
                phi,
                phi_torch,
            )
        )

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_multivariate_apply_rbf_feature_map(self):
        kernel_type = torchkernels.KernelType.RBF
        self._test_torch_v_numpy_apply_feature_map(kernel_type)

    @unittest.skipIf(SKIP, 'torch not available')
    def test_torch_v_numpy_multivariate_apply_delta_feature_map(self):
        kernel_type = torchkernels.KernelType.DELTA
        self._test_torch_v_numpy_apply_feature_map(kernel_type)

    def _test_torch_v_numpy_multivariate_apply_feature_map(self, kernel_type):
        print('\n...Running test of multivariate `apply_feature_map` on CPU v. GPU...')
        print(f'kernel_type: {kernel_type}')
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=10000, high=20000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        if kernel_type == torchkernels.KernelType.DELTA:
            m = np.random.randint(low=6, high=20)
            x = np.random.randint(m, size=(d, n))
        else:
            x = np.random.uniform(size=(d, n))
        x_torch = torch.from_numpy(x)
        assert x.shape == x_torch.size()
        phi: np.ndarray = kernels.apply_feature_map(
            kernel_type, x, l, batch_size, is_multivariate=True
        )
        phi_torch: np.ndarray = torchkernels.apply_feature_map(
            kernel_type, x_torch, l, batch_size, is_multivariate=True
        )
        gram_dim: int = num_batches * batch_size**2
        self.assertEqual(
            phi.shape,
            (gram_dim, 1)
        )
        self.assertEqual(
            phi_torch.shape,
            (gram_dim, 1)
        )
        self.assertTrue(
            np.allclose(
                phi,
                phi_torch,
            )
        )

    @unittest.skipIf(
        SKIP_CUDA, "Skipping test of GPU run because CUDA is not available")
    def test_torch_v_numpy_apply_rbf_feature_map_cuda(self):
        kernel_type = torchkernels.KernelType.RBF
        self._test_torch_v_numpy_apply_feature_map_cuda(kernel_type)

    @unittest.skipIf(
        SKIP_CUDA, "Skipping test of GPU run because CUDA is not available")
    def test_torch_v_numpy_apply_delta_feature_map_cuda(self):
        kernel_type = torchkernels.KernelType.DELTA
        self._test_torch_v_numpy_apply_feature_map_cuda(kernel_type)

    def _test_torch_v_numpy_apply_feature_map_cuda(self, kernel_type):
        print('\n...Running test of featwise `apply_feature_map` on CPU v. GPU...')
        print(f'kernel_type: {kernel_type}')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        d: int = np.random.randint(low=15, high=20)
        n: int = np.random.randint(low=50000, high=90000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 250
        batch_size = n // num_batches

        print(f'number of features: {d}')
        print(f'number of samples: {n}')
        print(f'number of batches: {num_batches}')
        print(f'batch size: {batch_size}')
        if kernel_type == torchkernels.KernelType.DELTA:
            m = np.random.randint(low=6, high=20)
            x = torch.randint(high=m, size=(d, n))
        else:
            x = torch.randn(size=(d, n), dtype=torch.float64)
        x_cuda = x.to(device)

        # cpu run
        t0 = datetime.datetime.now()
        phi: Tensor = torchkernels.apply_feature_map(
            kernel_type, x, l, batch_size
        )
        t1 = datetime.datetime.now()
        dt_cpu = t1 - t0
        cpu_runtime = dt_cpu.seconds + 1e-6 * dt_cpu.microseconds
        print(f'cpu runtime: {cpu_runtime} seconds')

        # cuda run
        t0 = datetime.datetime.now()
        phi_cuda: Tensor = torchkernels.apply_feature_map(
            kernel_type, x_cuda, l, batch_size
        )
        t1 = datetime.datetime.now()
        dt_gpu = t1 - t0
        gpu_runtime = dt_gpu.seconds + 1e-6 * dt_gpu.microseconds
        print(f'gpu runtime: {gpu_runtime} seconds')

        phi_cuda = phi_cuda.detach().cpu()

        self.assertEqual(
            phi.size(),
            phi_cuda.size()
        )
        self.assertEqual(
            phi.dtype,
            phi_cuda.dtype
        )
        self.assertTrue(
            torch.allclose(
                phi,
                phi_cuda,
                atol=1e-4,
                rtol=1e-5
            )
        )


if __name__ == '__main__':
    if SKIP:
        print('Torch could not be imported. I am skipping the tests')
    else:
        unittest.main()
