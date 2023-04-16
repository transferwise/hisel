import unittest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hisel import kernels


class KernelTest(unittest.TestCase):
    def test_rbf(self):
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=1000, high=2000)
        l: float = np.random.uniform(low=.95, high=5.)
        rbf = RBF(l)
        x = np.random.uniform(size=(d, n))
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

    def test_apply_feature_map(self):
        d: int = np.random.randint(low=2, high=10)
        n: int = np.random.randint(low=10000, high=15000)
        l: float = np.random.uniform(low=.95, high=5.)
        num_batches = 10
        batch_size = n // num_batches
        x = np.random.uniform(size=(d, n))
        phi: np.ndarray = kernels.apply_feature_map(
            x, l, batch_size
        )
        gram_dim: int = num_batches * batch_size**2
        self.assertEqual(
            phi.shape,
            (gram_dim, d)
        )
        phi_no_parallel = kernels.apply_feature_map(
            x, l, batch_size, no_parallel=True
        )
        self.assertEqual(
            phi_no_parallel.shape,
            (gram_dim, d)
        )
        self.assertTrue(
            np.allclose(
                phi,
                phi_no_parallel
            )
        )


if __name__ == '__main__':
    unittest.main()
