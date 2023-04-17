from hisel import kernels
import numpy as np


def hsic_b(
        x: np.ndarray,
        y: np.ndarray,
):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]
    n: int = x.shape[0]  # number of samples
    dx: int = x.shape[1]
    dy: int = y.shape[1]
    lx: float = np.sqrt(dx)
    ly: float = np.sqrt(dy)
    xgram: np.ndarray = kernels.multivariate_phi(
        x.T, lx
    )
    k = xgram[0, :, :]
    ygram: np.ndarray = kernels.multivariate_phi(
        y.T, ly
    )
    l = kernels._center_gram(ygram)[0, :, :]
    return np.trace(k @ l) / (n*n)
