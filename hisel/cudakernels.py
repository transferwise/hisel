from typing import Optional
from joblib import Parallel, delayed
from enum import Enum
import numpy as np
from tqdm import tqdm
from hisel.kernels import KernelType, Device

CUPY_AVAILABLE = True
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    print(f'Could not import cupy!')
    cp = np
    CUPY_AVAILABLE = False


def featwise(
        x: cp.ndarray,
        l: float,
        kernel_type: KernelType,
        catcont_split: Optional[int] = None
) -> cp.ndarray:
    if kernel_type == KernelType.RBF:
        return _rbf_featwise(x, l)
    elif kernel_type == KernelType.DELTA:
        return _delta_featwise(x)
    elif kernel_type == KernelType.BOTH:
        split = catcont_split if catcont_split else 0
        g_cat = _delta_featwise(x[:split, :].astype(int))
        g_cont = _rbf_featwise(x[split:, :], l)
        g = cp.concatenate((g_cat, g_cont), axis=0)
        return g
    else:
        raise ValueError(kernel_type)


def multivariate(
        x: cp.ndarray,
        l: float,
        kernel_type: KernelType,
        catcont_split: Optional[int] = None
) -> cp.ndarray:
    if kernel_type == KernelType.RBF:
        return _rbf_multivariate(x, l)
    elif kernel_type == KernelType.DELTA:
        return _delta_multivariate(x)
    elif kernel_type == KernelType.BOTH:
        split = catcont_split if catcont_split else 0
        g_cat = _delta_multivariate(x[:split, :].astype(int))
        g_cont = _rbf_multivariate(x[split:, :], l)
        g = cp.concatenate((g_cat, g_cont), axis=0)
        return g
    else:
        raise ValueError(kernel_type)


def _rbf_featwise(
        x: cp.ndarray,
        l: float
) -> cp.ndarray:
    assert x.ndim == 2
    d, n = x.shape
    z = cp.expand_dims(x, axis=2)
    s = cp.expand_dims(x, axis=1)
    s2 = cp.repeat(
        cp.square(s),
        repeats=n,
        axis=1,
    )
    z2 = cp.transpose(s2, (0, 2, 1))
    delta = z2 + s2 - 2*z @ s
    grams = cp.exp(-delta / (2*l*l))
    return grams


def _delta_featwise(
        x: cp.ndarray,
) -> cp.ndarray:
    assert x.ndim == 2
    d, n = x.shape
    assert x.dtype == int
    s = cp.expand_dims(x, axis=1)
    s2 = cp.repeat(
        s,
        repeats=n,
        axis=1,
    )
    z2 = cp.transpose(s2, (0, 2, 1))
    normalisation = cp.ones_like(s2)
    for i in range(d):
        cnt = cp.bincount(x[i, :])
        normalisation[i, :, :] = cnt[s2[i, :, :]]
    grams = cp.asarray(s2 == z2, dtype=float) / normalisation
    return grams


def _rbf_multivariate(
        x: cp.ndarray,
        l: float
) -> cp.ndarray:
    nx = x.shape[1]
    x2 = cp.tile(
        cp.sum(cp.square(x), axis=0),
        (nx, 1)
    )
    delta = x2.T + x2 - 2 * x.T @ x
    gram = cp.exp(-delta / (2 * l * l))
    return gram


def _rbf_hsic_b(
        x: cp.ndarray
) -> cp.ndarray:
    d, n = x.shape
    x2 = cp.cumsum(cp.square(x), axis=0)
    x2 = cp.expand_dims(x2, axis=1)
    x2 = cp.repeat(x2, n, axis=1)
    cross = cp.zeros(shape=(d, n, n))
    for i in range(d):
        cross[i] = x[:i+1, :].T @ x[:i+1, :]
    delta = x2.transpose(0, 2, 1) + x2 - 2 * cross
    ls2 = cp.arange(1, d+1).reshape(d, 1, 1)
    grams = cp.exp(-delta / (2 * ls2))
    return grams


def _delta_multivariate(
        x: cp.ndarray,
) -> cp.ndarray:
    assert x.dtype == int
    nx = x.shape[1]
    xmax = cp.roll(1 + cp.amax(x, axis=1, keepdims=True), 1)
    xmax[0, 0] = 1
    xflat = cp.sum(x * xmax, axis=0, keepdims=True)
    xx = cp.repeat(
        xflat,
        repeats=nx,
        axis=0
    )
    cnt = cp.bincount(xflat[0, :])
    normalisation = cnt[xx]
    gram = cp.asarray(xx == xx.T, dtype=float) / normalisation
    return gram


def _delta_hsic_b(
        x: cp.ndarray,
) -> cp.ndarray:
    d, n = x.shape
    grams = cp.empty(shape=(d, n, n), dtype=float)
    for i in range(d):
        grams[i, :, :] = _delta_multivariate(x[:i+1])
    return grams


def hsic_b(
        x: cp.ndarray,
        kernel_type: KernelType,
) -> cp.ndarray:
    if kernel_type == KernelType.DELTA:
        return _delta_hsic_b(x)
    else:
        return _rbf_hsic_b(x)


def multivariate_phi(
        x: cp.ndarray,
        l: float,
        kernel_type: KernelType,
        catcont_split: Optional[int] = None
) -> cp.ndarray:
    gram = multivariate(x, l, kernel_type, catcont_split)
    gram = cp.expand_dims(gram, axis=0)
    return gram


def _centering_matrix(d: int, n: int) -> cp.ndarray:
    id_ = cp.eye(n)
    ids = cp.repeat(cp.expand_dims(id_, axis=0), repeats=d, axis=0)
    ones = cp.ones_like(ids)
    h = ids - ones / n
    return h


def _center_gram_matmul(
        g: cp.ndarray,
        h: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if h is None:
        h = _centering_matrix(g.shape[0], g.shape[2])
    return h @ g @ h


def _center_gram(
        g: cp.ndarray,
) -> cp.ndarray:
    g -= cp.mean(g, axis=-1, keepdims=True)
    g -= cp.mean(g, axis=-2, keepdims=True)
    return g


def _run_batch(
        kernel_type: KernelType,
        x: cp.ndarray,
        l: float,
        is_multivariate: bool = False,
        catcont_split: Optional[int] = None,
) -> np.ndarray:
    phi = multivariate_phi if is_multivariate else featwise
    grams: cp.ndarray = _center_gram(phi(x, l, kernel_type, catcont_split))
    d, n, m = grams.shape
    assert n == m
    g_: cp.ndarray = cp.reshape(grams, (d, n*m)).T
    g: np.ndarray
    if CUPY_AVAILABLE:
        g = cp.asnumpy(g_)
    else:
        g = np.array(g_)
    return g


def _make_batches(x, batch_size):
    _, n = x.shape
    b = min(n, batch_size)
    num_batches = n // b
    if CUPY_AVAILABLE:
        # move array from GPU to GPU
        x = cp.array(x)
    batches = cp.split(x[:, :num_batches * b], num_batches, axis=1)
    return batches


def apply_feature_map(
        kernel_type: KernelType,
        x: np.ndarray,
        l: float,
        batch_size: int,
        is_multivariate: bool = False,
        catcont_split: Optional[int] = None,
        # Unused variable, only to keep consistency with the same functions in hisel.kernels
        device: Device = Device.GPU,
) -> np.ndarray:
    d, n = x.shape
    b = min(n, batch_size)
    batches = _make_batches(x, batch_size)
    num_of_batches = len(batches)
    partial_phis = [_run_batch(
        kernel_type,
        batch,
        l,
        is_multivariate,
        catcont_split,
    ) for batch in tqdm(batches)]
    phi: np.ndarray = np.vstack(partial_phis)
    return phi
