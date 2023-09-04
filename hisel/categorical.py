from typing import Optional, Tuple, Callable, Union, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import adjusted_mutual_info_score
from joblib import Parallel, delayed
from tqdm import tqdm


from hisel import permutohedron


def _discretise(
        y: np.ndarray,
        num_quantiles: int = 10,
) -> np.ndarray:
    assert y.ndim < 3
    qs = np.linspace(0 + 1. / num_quantiles, 1 - 1. /
                     num_quantiles, num=num_quantiles)

    def _build(cont):
        assert cont.ndim == 1
        discr = np.zeros(shape=cont.shape, dtype=int)
        threshold = np.amin(cont)
        for q in qs:
            quant = np.quantile(cont, q)
            if quant > threshold:
                threshold = quant
                discr += np.array(cont > threshold, dtype=int)
        return discr

    res = np.zeros(shape=y.shape)
    if y.ndim == 2:
        for d in range(y.shape[1]):
            res[:, d] = _build(y[:, d])
    else:
        res = _build(y)
    return res


def _preprocess_datatypes(
        y: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            if y[col].dtype == bool:
                y[col] = y[col].astype(int)
    elif y.dtypes == bool:
        y = y.astype(int)
    ydtypes = y.dtypes if isinstance(y, pd.DataFrame) else [y.dtypes]
    for dtype in ydtypes:
        assert dtype == int or dtype == float
    return y


@dataclass
class Selection:
    indexes: np.ndarray
    features: List[str]


def select(
        xdf: pd.DataFrame,
        ydf: Union[pd.DataFrame, pd.Series],
        num_permutations: Optional[int] = None,
        im_ratio: float = .05,
        max_iter: int = 1,
        parallel: bool = False,
        random_state: Optional[int] = None,
) -> Selection:
    print(f'Number of categorical features: {xdf.shape[1]}')
    xdf = _preprocess_datatypes(xdf)
    x = xdf.values
    ydf = _preprocess_datatypes(ydf)
    allfeatures: List[np.ndarray] = []

    if isinstance(ydf, pd.Series):
        if ydf.dtypes == float:
            y = _discretise(ydf.values)
        else:
            y = ydf.values
        allfeatures.append(
            search(
                x, y,
                num_permutations=num_permutations,
                im_ratio=im_ratio,
                max_iter=max_iter,
                parallel=parallel,
                random_state=random_state,
            )
        )
    else:
        for col in ydf.columns:
            if ydf[col].dtypes == float:
                y = _discretise(ydf[col].values)
            else:
                y = ydf[col].values
            allfeatures.append(
                search(
                    x, y,
                    num_permutations=num_permutations,
                    im_ratio=im_ratio,
                    max_iter=max_iter,
                    parallel=parallel,
                    random_state=random_state,
                )
            )
    fs = np.concatenate(allfeatures)
    indexes = np.array(list(set(fs)), dtype=int)
    features = list(xdf.columns[indexes])
    print(f'Number of selected categorical features: {len(features)}')
    return Selection(indexes=indexes, features=features)


def search(
        x: np.ndarray,
        y: np.ndarray,
        num_permutations: Optional[int] = None,
        im_ratio: float = .05,
        max_iter: int = 1,
        parallel: bool = False,
        random_state: Optional[int] = None,
) -> np.ndarray:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    n, d = x.shape
    assert x.dtype == int
    assert y.dtype == int
    if num_permutations is None:
        num_permutations = 1
    x = x - np.amin(x, axis=0, keepdims=True)
    y = y - np.amin(y, axis=0, keepdims=True)
    active_set = set(range(d))
    sel = np.arange(d, dtype=int)
    features = np.array([], dtype=int)
    imall = .0
    n_iter = 0
    while len(active_set) > 0 and n_iter < max_iter:
        active = np.array(list(active_set))
        num_active = len(active)
        num_haar_samples = min(
            max(1, num_permutations // num_active),
            2**num_active // num_active
        )
        permutations = permutohedron.haar_sampling(
            num_active,
            size=num_haar_samples,
            random_state=random_state
        )
        if parallel:
            tries = Parallel(n_jobs=-1)([
                delayed(_try_permutation)(
                    ami, x, y, active, list(permutation))
                for permutation in tqdm(permutations)
            ])
        else:
            tries = [_try_permutation(
                ami, x, y, active, list(permutation)) for permutation in tqdm(permutations)]

        im = .0
        for im_, sel_ in tries:
            if im_ > im:
                sel = sel_
                im = im_
        if im < im_ratio * imall:
            print('im < im_ratio * imall')
            print(f'{im} < {im_ratio} * {imall}')
            break
        elif im > imall:
            imall = im

        features = np.concatenate((features, sel))
        active_set = active_set.difference(set(features))
        n_iter += 1
    threshold = im_ratio * imall
    fwsel = _featurewise_selection(
        ami,
        x,
        y,
        threshold
    )
    features = np.array(list(
        set(features).union(set(fwsel))
    ))
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


def _featurewise_selection(
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        threshold: float,
) -> List[int]:
    sel = []
    for i in range(x.shape[1]):
        v = metric(x[:, [i]], y)
        if v > threshold:
            sel.append(i)
    return sel
