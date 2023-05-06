from typing import Optional, Set, List
from hisel import kernels
from hisel.kernels import KernelType
from scipy.stats import special_ortho_group
import numpy as np
from joblib import Parallel, delayed


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


def search(
        x: np.ndarray,
        y: np.ndarray,
        xkerneltype: Optional[KernelType] = None,
        ykerneltype: Optional[KernelType] = None,
        num_permutations: int = 3,
        im_ratio: float = .1,
        parallel: bool = True,
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
    ygram: np.ndarray = kernels.multivariate_phi(
        y.T, ly, ykerneltype
    )
    l = kernels._center_gram(ygram)
    xt = x.T
    active = set(range(dx))
    features = set({})
    im = .0
    while len(active) > 1:
        a = np.array(list(active))
        len_a = len(a)
        hsim = np.zeros(shape=(len_a,))
        res = []
        permutations = set()
        m = min(num_permutations, 2**len_a // len_a)
        for rs in range(m):
            permutations.update(_sample_permutations(len_a, random_state=rs))
        if parallel:
            res = Parallel(n_jobs=-1)([
                delayed(_try_permutation)(
                    xt, l, xkerneltype, a, list(permutation))
                for permutation in permutations
            ])
        else:
            res = [_try_permutation(
                xt, l, xkerneltype, a, list(permutation))
                for permutation in permutations
            ]

        for hsim_, idx_ in res:
            if np.amax(hsim_) > np.amax(hsim):
                hsim = hsim_
                idx = idx_
                s = np.argmax(hsim)

        s = np.argmax(hsim)
        if hsim[s] < im_ratio * im:
            break
        elif hsim[s] > im:
            im = hsim[s]
        features = features.union(set(idx[:s+1]))
        active = active.difference(features)
    return features


def _try_permutation(
        xt: np.ndarray,
        l: np.ndarray,
        xkerneltype: KernelType,
        active: np.ndarray,
        permutation: np.ndarray,
):
    selection = active[permutation]
    xgrams = kernels.hsic_b(xt[selection, :], xkerneltype)
    hsim = np.trace(xgrams @ l, axis1=1, axis2=2)
    return hsim, selection


def _sample_permutations(
        d,
        random_state: Optional[int] = None,
):
    def projection(d: int):
        p = np.diag(np.arange(-1, -d, -1, dtype=float), 1)
        p += np.eye(d)
        for k in range(1, d):
            p += np.diag(np.ones(shape=d-k), -k)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p
    u = special_ortho_group.rvs(d, random_state=random_state)[:, :d-1]
    xs = np.concatenate((u, -u), axis=1)
    p = projection(d)
    perms = np.argsort(p.T @ xs, axis=0)
    permutations = set([
        tuple(sigma) for sigma in perms.T])
    return permutations
