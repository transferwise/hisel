import unittest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hisel import kernels
import datetime


pyhsiclasso_recon = True


class KernelTest(unittest.TestCase):

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_gaussian_recon(self):
        kernel_type = kernels.KernelType.RBF
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        x = np.random.uniform(size=(d, n))
        x /= np.std(x) + 1e-19
        f = kernels.featwise(x, l, kernel_type)
        p = kernels.pyhsiclasso_kernel_gaussian(x, x, l)
        f_recon = f[0, :, :]
        self.assertEqual(
            f_recon.shape,
            p.shape
        )
        self.assertTrue(
            np.allclose(
                f_recon,
                p
            )
        )

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_delta_recon(self):
        kernel_type = kernels.KernelType.DELTA
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        m: int = np.random.randint(low=6, high=12)
        x = np.random.randint(m, size=(d, n))
        f = kernels.featwise(x, l, kernel_type)
        p = kernels.pyhsiclasso_kernel_delta_norm(x, x)
        f_recon = f[0, :, :]
        self.assertEqual(
            f_recon.shape,
            p.shape
        )
        self.assertTrue(
            np.allclose(
                f_recon,
                p
            )
        )

    def test_rbf(self):
        print(f'\n...Test RBF...')
        kernel_type = kernels.KernelType.RBF
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        rbf = RBF(l)
        x = np.random.uniform(size=(d, n))
        x /= np.std(x, axis=1, keepdims=True) + 1e-19
        k = np.zeros((d, n, n))
        g = np.zeros((d, n, n))
        for i in range(d):
            k[i, :, :] = rbf(x[[i], :].T)
            g[i, :, :] = kernels.multivariate(
                x[[i], :], x[[i], :], l, kernel_type)

        f = kernels.featwise(x, l, kernel_type)
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

    def test_delta(self):
        print(f'\n...Test DELTA...')
        kernel_type = kernels.KernelType.DELTA
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l = 1.
        m: int = np.random.randint(low=6, high=12)
        x = np.random.randint(m, size=(d, n))
        g = np.zeros((d, n, n))
        for i in range(d):
            g[i, :, :] = kernels.multivariate(
                x[[i], :], x[[i], :], l, kernel_type)

        f = kernels.featwise(x, l, kernel_type)

        self.assertTrue(
            np.allclose(
                g, f
            )
        )

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_recon_rbf(self):
        print(f'\n...Recon with pyHSICLasso...')
        kernel_type = kernels.KernelType.RBF
        print(f'kernel_type: {kernel_type}')
        # Notice the two normalisation that pyHSICLasso uses
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        x = np.random.uniform(size=(d, n))
        x /= np.std(x, axis=1, keepdims=True) + 1e-19  # normalisation n.1
        f = kernels._center_gram(kernels.featwise(x, l, kernel_type))
        f_recon = np.array(f, copy=True)
        p = np.zeros((d, n*n))
        for i in range(d):
            fronorm = np.linalg.norm(f_recon[i, :, :], 'fro') + 1e-9
            p[i, :] = kernels.pyhsiclasso_compute_kernel(
                x[[i], :], kernel_type)
            f_recon[i, :, :] /= fronorm  # normalisation n.2
        f_recon = f_recon.reshape(d, n*n)
        self.assertEqual(
            f_recon.shape,
            p.shape
        )
        self.assertTrue(
            np.allclose(
                f_recon, p,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_recon_delta(self):
        print(f'\n...Recon with pyHSICLasso...')
        kernel_type = kernels.KernelType.DELTA
        print(f'kernel_type: {kernel_type}')
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        m: int = np.random.randint(low=6, high=12)
        x = np.random.randint(m, size=(d, n))
        f = kernels._center_gram(kernels.featwise(x, l, kernel_type))
        f_recon = np.array(f, copy=True)
        p = np.zeros((d, n*n))
        for i in range(d):
            fronorm = np.linalg.norm(f_recon[i, :, :], 'fro') + 1e-9
            p[i, :] = kernels.pyhsiclasso_compute_kernel(
                x[[i], :], kernel_type)
            f_recon[i, :, :] /= fronorm  # normalisation n.2
        f_recon = f_recon.reshape(d, n*n)
        self.assertEqual(
            f_recon.shape,
            p.shape
        )
        self.assertTrue(
            np.allclose(
                f_recon, p,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_centering_matrix(self):
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        h: np.ndarray = kernels._centering_matrix(d, n)
        self.assertEqual(
            h.shape,
            (d, n, n)
        )
        h_: np.ndarray = np.eye(n) - np.ones(n) / n
        self.assertTrue(
            np.allclose(
                h_, h[0, :, :]
            )
        )

    def test_multivariate_rbf(self):
        # When the input is one dimensional,
        # computing a multivariate kernel and a feature-wise kernel
        # must yield the same result
        kernel_type = kernels.KernelType.RBF
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        x = np.random.uniform(size=(d, n))
        gu = kernels._run_batch(kernel_type, x, l, is_multivariate=False)
        gm = kernels._run_batch(kernel_type, x, l, is_multivariate=True)
        self.assertEqual(
            gu.shape,
            gm.shape
        )
        self.assertTrue(
            np.allclose(
                gm,
                gu,
            )
        )

    def test_multivariate_delta(self):
        # When the input is one dimensional,
        # computing a multivariate kernel and a feature-wise kernel
        # must yield the same result
        kernel_type = kernels.KernelType.DELTA
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        x = np.random.randint(10, size=(d, n))
        gu = kernels._run_batch(kernel_type, x, l, is_multivariate=False)
        gm = kernels._run_batch(kernel_type, x, l, is_multivariate=True)
        self.assertEqual(
            gu.shape,
            gm.shape
        )
        self.assertTrue(
            np.allclose(
                gm,
                gu,
            )
        )

    # @unittest.skip
    def test_apply_rbf_feature_map(self):
        kernel_type = kernels.KernelType.RBF
        self._test_apply_feature_map(kernel_type)

    # @unittest.skip
    def test_apply_delta_feature_map(self):
        kernel_type = kernels.KernelType.DELTA
        self._test_apply_feature_map(kernel_type)

    def _test_apply_feature_map(self, kernel_type):
        print(f'\n...Test apply_feature_map...')
        print(f'kernel_type: {kernel_type}')
        d: int = np.random.randint(low=5, high=10)
        n: int = np.random.randint(low=20000, high=35000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        if kernel_type == kernels.KernelType.DELTA:
            x = np.random.randint(10, size=(d, n))
        else:
            x = np.random.uniform(size=(d, n))

        # Execution with parallelization enabled
        t0 = datetime.datetime.now()
        phi: np.ndarray = kernels.apply_feature_map(
            kernel_type, x, l, batch_size
        )
        t1 = datetime.datetime.now()
        dt_parallel = t1 - t0
        parallel_runtime = dt_parallel.seconds + 1e-6 * dt_parallel.microseconds
        gram_dim: int = num_batches * batch_size**2
        self.assertEqual(
            phi.shape,
            (gram_dim, d)
        )
        print(f'runtime with parallel execution: {parallel_runtime} seconds')

        # Execution with parallelization disabled
        t0 = datetime.datetime.now()
        phi_no_parallel = kernels.apply_feature_map(
            kernel_type, x, l, batch_size, no_parallel=True
        )
        t1 = datetime.datetime.now()
        dt_serial = t1 - t0
        serial_runtime = dt_serial.seconds + 1e-6 * dt_serial.microseconds

        self.assertEqual(
            phi_no_parallel.shape,
            (gram_dim, d)
        )
        print(f'runtime with serial execution: {serial_runtime} seconds')

        # check that serial and parallel execution yield the same results
        self.assertTrue(
            np.allclose(
                phi,
                phi_no_parallel
            )
        )


if __name__ == '__main__':
    unittest.main()
