from typing import Optional, Set, Tuple, Callable, Union, List
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from joblib import Parallel, delayed


import permutohedron


def select_features(
        x: np.ndarray,
        y: np.ndarray,
        im_ratio: float = .05,
        num_haar_samples: int = 1,
        parallel: bool = False,
        max_iter: int = 1,
) -> np.ndarray:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    n, d = x.shape
    assert x.dtype == int
    assert y.dtype == int
    x = x - np.amin(x, axis=0, keepdims=True)
    y = y - np.amin(y, axis=0, keepdims=True)
    active_set = set(range(d))
    sel = np.arange(d, dtype=int)
    features = np.array([], dtype=int)
    imall = .0
    random_state = None
    n_iter = 0
    while len(active_set) > 1 and n_iter < max_iter:
        active = np.array(list(active_set))
        num_active = len(active)
        num_haar_samples = min(num_haar_samples, 2**num_active // num_active)
        permutations = permutohedron.haar_sampling(
            num_active,
            size=num_haar_samples,
            random_state=random_state
        )
        if parallel:
            tries = Parallel(n_jobs=-1)([
                delayed(_try_permutation)(
                    ami, x, y, active, list(permutation))
                for permutation in permutations
            ])
        else:
            tries = [_try_permutation(
                ami, x, y, active, list(permutation)) for permutation in permutations]

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


def ami(
        x: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:
    n, d = x.shape
    assert n == y.shape[0]
    z = _encode(x)
    im = np.empty(shape=(d, ), dtype=float)
    for i in range(d):
        im[i] = adjusted_mutual_info_score(z[:, i], y)
    return im


def _encode(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2
    ns = 1 + np.amax(x, axis=0, keepdims=True)
    res = np.array(x, copy=True)
    ms = np.roll(ns, 1, axis=1)
    ms[0, 0] = 1
    ms = np.cumprod(ms, axis=1)
    res = np.cumsum(ms * x, axis=1)
    return res


def _try_permutation(
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        active: np.ndarray,
        permutation: Union[List[int], np.ndarray],
) -> Tuple[float, np.ndarray]:
    sel = active[permutation]
    ims = metric(x[:, sel], y)
    s = np.argmax(ims)
    im = ims[s]
    selection = sel[:s+1]
    return im, selection
