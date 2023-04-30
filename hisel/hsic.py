from typing import Optional
from hisel import kernels
from hisel.kernels import KernelType
import numpy as np


def hsic_b(
        x: np.ndarray,
        y: np.ndarray,
        xkerneltype: Optional[KernelType] = None,
        ykerneltype: Optional[KernelType] = None,
):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]
    n: int = x.shape[0]  # number of samples
    dx: int = x.shape[1]
    dy: int = y.shape[1]
    lx: float = np.sqrt(dx)
    ly: float = np.sqrt(dy)
    if xkerneltype is None:
        if x.dtype == int:
            xkerneltype = KernelType.DELTA
        else:
            xkerneltype = KernelType.RBF
    if ykerneltype is None:
        if y.dtype == int:
            ykerneltype = KernelType.DELTA
        else:
            ykerneltype = KernelType.RBF
    xgram: np.ndarray = kernels.multivariate_phi(
        x.T, lx, xkerneltype
    )
    k = xgram[0, :, :]
    ygram: np.ndarray = kernels.multivariate_phi(
        y.T, ly, ykerneltype
    )
    l = kernels._center_gram(ygram)[0, :, :]
    return np.trace(k @ l) / (n*n)
