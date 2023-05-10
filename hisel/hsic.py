from typing import Optional, List, Union, Set, Tuple
from hisel import kernels
from hisel.kernels import KernelType
from hisel import permutohedron
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
        num_permutations: Optional[int] = None,
        im_ratio: float = .1,
        max_iter: int = 3,
        parallel: bool = True,
        random_state: Optional[int] = None,
):
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == y.shape[0]
    dx: int = x.shape[1]
    dy: int = y.shape[1]
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
    if num_permutations is None:
        num_permutations = 3 * dx
    ygram: np.ndarray = kernels.multivariate_phi(
        y.T, ly, ykerneltype
    )
    l = kernels._center_gram(ygram)
    xt = x.T
    active_set = set(range(dx))
    sel = np.arange(d, dtype=int)
    features = np.array([], dtype=int)
    imall = .0
    n_iter = 0
    while len(active) > 1 and n_iter < max_iter:
        active = np.array(list(active_set))
        num_active = len(a)
        num_haar_samples = min(
            max(1, num_permutations // num_active),
            2**num_active // num_active
        )
        permutations = _sample_permutations(
            d, size=num_haar_samples, random_state=random_state)
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

        im = .0
        for im_, sel_ in tries:
            if im_ > im:
                sel = sel_
                im = im_
        if im < im_ratio * imall:
            break
        elif im > imall:
            imall = im

        features = np.concatenate((features, sel))
        active_set = active_set.difference(set(features))
        n_iter += 1
    return features


def _try_permutation(
        xt: np.ndarray,
        l: np.ndarray,
        xkerneltype: KernelType,
        active: np.ndarray,
        permutation: Union[List[int], np.ndarray],
):
    selection = active[permutation]
    xgrams = kernels.hsic_b(xt[selection, :], xkerneltype)
    hsim = np.trace(xgrams @ l, axis1=1, axis2=2)
    s = np.argmax(hsim)
    selection = selection[:s+1]
    im = hsim[s]
    return im, selection


def _sample_permutations(
        d: int,
        size: int = 1,
        random_state: Optional[int] = None,
) -> Set[Tuple[int, ...]]:
    return permutohedron.haar_sampling(
        d, size, random_state)
