import unittest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hisel import kernels
import datetime


pyhsiclasso_recon = True


class KernelTest(unittest.TestCase):

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_gaussian_recon(self):
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        x = np.random.uniform(size=(d, n))
        x /= np.std(x) + 1e-19
        f = kernels.featwise(x, l)
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

    def test_rbf(self):
        print(f'\n...Test RBF...')
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
            g[i, :, :] = kernels.multivariate(x[[i], :], x[[i], :], l)

        f = kernels.featwise(x, l)
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

    @unittest.skipIf(not pyhsiclasso_recon, 'Skipping reconciliation with pyHSICLasso')
    def test_pyhsiclasso_kernel_recon(self):
        print(f'\n...Recon with pyHSICLasso...')
        # Notice the two normalisation that pyHSICLasso uses
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = 1.
        rbf = RBF(l)
        x = np.random.uniform(size=(d, n))
        x /= np.std(x, axis=1, keepdims=True) + 1e-19  # normalisation n.1
        f = kernels._center_gram(kernels.featwise(x, l))
        f_recon = np.array(f, copy=True)
        p = np.zeros((d, n*n))
        for i in range(d):
            fronorm = np.linalg.norm(f_recon[i, :, :], 'fro') + 1e-9
            p[i, :] = kernels.pyhsiclasso_compute_kernel(
                x[[i], :])
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

    def test_multivariate(self):
        # When the input is one dimensional,
        # computing a multivariate kernel and a feature-wise kernel
        # must yield the same result
        d: int = 1
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        x = np.random.uniform(size=(d, n))
        gu = kernels._run_batch(x, l, is_multivariate=False)
        gm = kernels._run_batch(x, l, is_multivariate=True)
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

    @unittest.skip
    def test_apply_feature_map(self):
        print(f'\n...Test apply_feature_map...')
        d: int = np.random.randint(low=5, high=10)
        n: int = np.random.randint(low=20000, high=35000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        x = np.random.uniform(size=(d, n))

        # Execution with parallelization enabled
        t0 = datetime.datetime.now()
        phi: np.ndarray = kernels.apply_feature_map(
            x, l, batch_size
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
            x, l, batch_size, no_parallel=True
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
