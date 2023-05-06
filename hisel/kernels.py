from typing import Optional
from joblib import Parallel, delayed
from enum import Enum
import numpy as np


class KernelType(Enum):
    RBF = 0
    DELTA = 1


def featwise(
        x: np.ndarray,
        l: float,
        kernel_type: KernelType,
) -> np.ndarray:
    if kernel_type == KernelType.RBF:
        return _rbf_featwise(x, l)
    elif kernel_type == KernelType.DELTA:
        return _delta_featwise(x)
    else:
        raise ValueError(kernel_type)


def multivariate(
        x: np.ndarray,
        y: np.ndarray,
        l: float,
        kernel_type: KernelType,
) -> np.ndarray:
    if kernel_type == KernelType.RBF:
        return _rbf_multivariate(x, y, l)
    elif kernel_type == KernelType.DELTA:
        # Notice that only x is considered in the delta case
        return _delta_multivariate(x)
    else:
        raise ValueError(kernel_type)


def _rbf_featwise(
        x: np.ndarray,
        l: float
) -> np.ndarray:
    assert x.ndim == 2
    d, n = x.shape
    z = np.expand_dims(x, axis=2)
    s = np.expand_dims(x, axis=1)
    s2 = np.repeat(
        np.square(s),
        repeats=n,
        axis=1,
    )
    z2 = np.transpose(s2, (0, 2, 1))
    delta = z2 + s2 - 2*z @ s
    grams = np.exp(-delta / (2*l*l))
    return grams


def _delta_featwise(
        x: np.ndarray,
) -> np.ndarray:
    assert x.ndim == 2
    d, n = x.shape
    assert x.dtype == int
    s = np.expand_dims(x, axis=1)
    s2 = np.repeat(
        s,
        repeats=n,
        axis=1,
    )
    z2 = np.transpose(s2, (0, 2, 1))
    normalisation = np.ones_like(s2)
    for i in range(d):
        cnt = np.bincount(x[i, :])
        normalisation[i, :, :] = cnt[s2[i, :, :]]
    grams = np.asarray(s2 == z2, dtype=float) / normalisation
    return grams


def _rbf_multivariate(
        x: np.ndarray,
        y: np.ndarray,
        l: float
) -> np.ndarray:
    nx = x.shape[1]
    ny = y.shape[1]
    x2 = np.tile(
        np.sum(np.square(x), axis=0),
        (ny, 1)
    )
    y2 = np.tile(
        np.sum(np.square(y), axis=0),
        (nx, 1)
    )
    delta = x2.T + y2 - 2 * x.T @ y
    gram = np.exp(-delta / (2 * l * l))
    return gram


def _rbf_hsic_b(
        x: np.ndarray
) -> np.ndarray:
    d, n = x.shape
    x2 = np.cumsum(np.square(x), axis=0)
    x2 = np.expand_dims(x2, axis=1)
    x2 = np.repeat(x2, n, axis=1)
    cross = np.zeros(shape=(d, n, n))
    for i in range(d):
        cross[i] = x[:i+1, :].T @ x[:i+1, :]
    delta = x2.transpose(0, 2, 1) + x2 - 2 * cross
    ls2 = np.arange(1, d+1).reshape(d, 1, 1)
    grams = np.exp(-delta / (2 * ls2))
    return grams


def _delta_multivariate(
        x: np.ndarray,
) -> np.ndarray:
    assert x.dtype == int
    nx = x.shape[1]
    xmax = np.roll(1 + np.amax(x, axis=1, keepdims=True), 1)
    xmax[0, 0] = 1
    xflat = np.sum(x * xmax, axis=0, keepdims=True)
    xx = np.repeat(
        xflat,
        repeats=nx,
        axis=0
    )
    cnt = np.bincount(xflat[0, :])
    normalisation = cnt[xx]
    gram = np.asarray(xx == xx.T, dtype=float) / normalisation
    return gram


def _delta_hsic_b(
        x: np.ndarray,
) -> np.ndarray:
    d, n = x.shape
    grams = np.empty(shape=(d, n, n), dtype=float)
    for i in range(d):
        grams[i, :, :] = _delta_multivariate(x[:i+1])
    return grams


def hsic_b(
        x: np.ndarray,
        kernel_type: KernelType,
) -> np.ndarray:
    if kernel_type == KernelType.DELTA:
        return _delta_hsic_b(x)
    else:
        return _rbf_hsic_b(x)


def multivariate_phi(
        x: np.ndarray,
        l: float,
        kernel_type: KernelType,
) -> np.ndarray:
    gram = multivariate(x, x, l, kernel_type)
    gram = np.expand_dims(gram, axis=0)
    return gram


def _centering_matrix(d: int, n: int) -> np.ndarray:
    id_ = np.eye(n)
    ids = np.repeat(np.expand_dims(id_, axis=0), repeats=d, axis=0)
    ones = np.ones_like(ids)
    h = ids - ones / n
    return h


def _center_gram(
        g: np.ndarray,
        h: Optional[np.ndarray] = None
) -> np.ndarray:
    if h is None:
        h = _centering_matrix(g.shape[0], g.shape[2])
    return h @ g @ h


def _run_batch(
        kernel_type: KernelType,
        x: np.ndarray,
        l: float,
        h: Optional[np.ndarray] = None,
        is_multivariate: bool = False,
) -> np.ndarray:
    phi = multivariate_phi if is_multivariate else featwise
    grams: np.ndarray = _center_gram(phi(x, l, kernel_type), h)
    d, n, m = grams.shape
    assert n == m
    g: np.ndarray = np.reshape(grams, (d, n*m)).T
    return g


def _make_batches(x, batch_size):
    _, n = x.shape
    b = min(n, batch_size)
    num_batches = n // b
    batches = np.split(x[:, :num_batches * b], num_batches, axis=1)
    return batches


def _can_allocate(d: int, n: int, num_batches: int):
    try:
        np.zeros((num_batches, d, n, n))
    except np.core._exceptions._ArrayMemoryError as e:
        print('Number of features and number of samples are too big to allocate the feature map.'
              'Reduce the number of samples and try again')
        raise(e)


def apply_feature_map(
        kernel_type: KernelType,
        x: np.ndarray,
        l: float,
        batch_size: int,
        is_multivariate: bool = False,
        no_parallel: bool = True
) -> np.ndarray:
    d, n = x.shape
    b = min(n, batch_size)
    batches = _make_batches(x, batch_size)
    num_of_batches = len(batches)
    # _can_allocate(d, n, num_of_batches)
    if no_parallel or num_of_batches < 2 or d*n < 100000:
        h = _centering_matrix(
            d, b) if not is_multivariate else _centering_matrix(1, b)
        partial_phis = [_run_batch(
            kernel_type,
            batch,
            l,
            h,
            is_multivariate
        ) for batch in batches]
    else:
        partial_phis = Parallel(n_jobs=-1)([
            delayed(_run_batch)(kernel_type, batch, l) for batch in batches
        ])
    phi: np.ndarray = np.vstack(partial_phis)
    return phi


##################################
# Reconciliation with pyHSICLasso
##################################


def pyhsiclasso_kernel_gaussian(X_in_1, X_in_2, sigma):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    X_in_12 = np.sum(np.power(X_in_1, 2), 0)
    X_in_22 = np.sum(np.power(X_in_2, 2), 0)
    dist_2 = np.tile(X_in_22, (n_1, 1)) + \
        np.tile(X_in_12, (n_2, 1)).transpose() - 2 * np.dot(X_in_1.T, X_in_2)
    K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
    return K


def pyhsiclasso_kernel_delta_norm(X_in_1, X_in_2):
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


def pyhsiclasso_compute_kernel(x, kernel_type: KernelType):
    d, n = x.shape
    B = n

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)

    if kernel_type == KernelType.RBF:

        x = (x / (x.std() + 10e-20)).astype(np.float32)

        k = pyhsiclasso_kernel_gaussian(x, x, np.sqrt(d))

    else:
        k = pyhsiclasso_kernel_delta_norm(x, x)

    k = np.dot(np.dot(H, k), H)

    # Normalize HSIC tr(k*k) = 1
    fronorm = np.linalg.norm(k, 'fro') + 10e-10
    k = k / fronorm
    return k.flatten()
#################################################
