# API
from typing import List, Optional, Union, Tuple, Sequence
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from hisel import lar, kernels, cudakernels
from hisel.kernels import KernelType, Device
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


class FeatureType(Enum):
    CONT = 0
    DISCR = 1
    BOTH = 2


def _choose_kernel(feature_type: FeatureType) -> KernelType:
    if feature_type is None:
        return KernelType.RBF
    elif feature_type == FeatureType.CONT:
        return KernelType.RBF
    elif feature_type == FeatureType.DISCR:
        return KernelType.DELTA
    elif feature_type == FeatureType.BOTH:
        return KernelType.BOTH
    return KernelType.RBF


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


def ksgmi(
        x: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        threshold: float = .01,
) -> Tuple[List[str], pd.Series]:
    x = _preprocess_datatypes(x)
    y = _preprocess_datatypes(y)
    discrete_features = x.dtypes == int
    mix = x.values
    if isinstance(y, pd.Series) or (isinstance(y, pd.DataFrame) and y.shape[1] == 1):
        miys = np.squeeze(y.values).reshape(-1, 1)
    else:
        miys = y.values
    sel = set()
    totmis = np.zeros(x.shape[1], dtype=float)
    for j in range(miys.shape[1]):
        miy = miys[:, j]
        compute_mi = mutual_info_classif if miy.dtype == int else mutual_info_regression
        mis = pd.Series(
            compute_mi(mix, miy, discrete_features=discrete_features),
            index=x.columns)
        mis /= np.max(mis)
        sel = sel.union(set(
            set(mis.loc[mis > threshold].index)
        ))
        totmis += mis
    mutual_infos = pd.Series(totmis, index=x.columns)
    relevant_features = list(sel)
    print(
        f'ksg-mi preprocessing: {len(relevant_features)} features are pre-selected')
    return relevant_features, mutual_infos


class HSICSelector:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 xfeattype: Optional[FeatureType] = None,
                 yfeattype: Optional[FeatureType] = None,
                 feature_names: Optional[List[str]] = None,
                 catcont_split: Optional[int] = None,
                 ):
        assert x.ndim == 2
        assert y.ndim == 2
        nx, dx = x.shape
        ny, dy = y.shape
        if xfeattype is None:
            xfeattype = FeatureType.DISCR if x.dtype in (
                np.int16, np.int32, np.int64, int) else FeatureType.CONT
        if yfeattype is None:
            yfeattype = FeatureType.DISCR if y.dtype in (
                np.int16, np.int32, np.int64, int) else FeatureType.CONT
        print('\nHSIC feature selection')
        print(f'Feature type of x: {xfeattype}')
        print(f'Feature type of y: {yfeattype}')
        print(f'Data type of x: {x.dtype}')
        print(f'Data type of y: {y.dtype}')
        print(f'Total number of features: {dx}')
        print(f'Dimensionality of target: {dy}')
        print(f'Number of x samples: {nx}')
        print(f'Number of y samples: {ny}')
        assert nx == ny, 'number of samples in x and in y must be equal'
        self.total_number_of_features = x.shape[1]
        self.x = np.array(x, copy=True)
        self.y = np.array(y, copy=True)
        self.xfeattype = xfeattype
        self.yfeattype = yfeattype
        self.xkerneltype = _choose_kernel(xfeattype)
        self.ykerneltype = _choose_kernel(yfeattype)
        self.catcont_split = catcont_split
        if feature_names is None:
            self.feature_names = [f'f{f}' for f in range(x.shape[1])]
            pass
        else:
            self.feature_names = feature_names

    def lasso_path(self):
        if not hasattr(self, 'lassopaths'):
            print(
                'You need to call the method `select` before accessing the lasso paths of the latest selection')
            raise ValueError()
        maxlen = max([p.shape[0] for p in self.lassopaths])
        paths: List[np.ndarray] = []
        for p in self.lassopaths:
            _p = np.zeros(shape=(1, maxlen, p.shape[1]), dtype=float)
            _p[0, :p.shape[0], :] = p
            _p[0, p.shape[0]:, :] = p[-1, :]
            paths.append(_p)
        path = np.mean(np.vstack(paths), axis=0)
        df: pd.DataFrame = pd.DataFrame(
            path, columns=self.feature_names)
        return df

    def projection_matrix(self,
                          number_of_features: int,
                          batch_size: int = 1000,
                          minibatch_size: int = 200,
                          number_of_epochs: int = 1,
                          device: Device = Device.CPU,
                          ) -> np.ndarray:
        p: np.ndarray = np.zeros(
            (number_of_features, self.total_number_of_features))
        features: List[int]
        lassopaths: List[np.ndarray] = []
        lassopath: np.ndarray
        x_, y_ = preprocess(
            x=self.x,
            xfeattype=self.xfeattype,
            y=self.y,
            yfeattype=self.yfeattype,
            repeat=1,
            standard=True,
            catcont_split=self.catcont_split,
        )
        xs = _make_batches(x_, batch_size)
        ys = _make_batches(y_, batch_size)
        for x, y in zip(xs, ys):
            x, y = preprocess(
                x=x,
                xfeattype=self.xfeattype,
                y=y,
                yfeattype=self.yfeattype,
                repeat=number_of_epochs,
                standard=False,
                catcont_split=self.catcont_split,
            )
            features, lassopath = _run(
                x,
                y,
                number_of_features,
                minibatch_size,
                self.xkerneltype,
                self.ykerneltype,
                catcont_split=self.catcont_split,
                device=device,
            )
            p += _to_projection_matrix(
                features,
                self.total_number_of_features,
                number_of_features,
            )
            lassopaths.append(lassopath)
        p /= len(xs)
        self.lassopaths = lassopaths
        return p

    def select(self,
               number_of_features: int,
               batch_size: int = 10000,
               minibatch_size: int = 200,
               number_of_epochs: int = 1,
               device: Device = Device.CPU,
               return_index: bool = False
               ) -> Union[Sequence[int], List[str]]:
        p = self.projection_matrix(
            number_of_features=number_of_features,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            number_of_epochs=number_of_epochs,
            device=device,
        )
        indexes = _to_feature_list(p)
        if return_index:
            return indexes
        features = list(
            np.array(self.feature_names)[indexes])
        return features

    def regularization_curve(self,
                             batch_size: int = 1000,
                             minibatch_size: int = 200,
                             number_of_epochs: int = 1,
                             device: Device = Device.CPU,
                             ):
        number_of_features = self.total_number_of_features - 1
        features = self.select(
            number_of_features,
            batch_size,
            minibatch_size,
            number_of_epochs,
            device,
            return_index=True,
        )
        path = self.lasso_path()
        curve = np.cumsum(np.sort(path.iloc[-1, :])[::-1])
        self.ordered_features = sorted(
            features,
            key=lambda a: path.iloc[-1, a],
            reverse=True
        )
        return curve

    def autoselect(self,
                   batch_size: int = 1000,
                   minibatch_size: int = 200,
                   number_of_epochs: int = 1,
                   threshold: float = .01,
                   device: Device = Device.CPU,
                   lasso_path: Optional[pd.DataFrame] = None,
                   ) -> List[str]:
        if lasso_path is None:
            curve = self.regularization_curve(
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                number_of_epochs=number_of_epochs,
                device=device,
            )
            lasso_path = self.lasso_path()
        return HSICSelector.select_from_lasso_path(lasso_path, threshold)

    @staticmethod
    def select_from_lasso_path(
        lasso_path: pd.DataFrame,
        threshold: float = .01,
    ) -> List[str]:
        features = list(lasso_path.columns)
        curve = np.cumsum(np.sort(lasso_path.iloc[-1, :])[::-1])
        ordered_features = sorted(
            features,
            key=lambda a: lasso_path[a].values[-1],
            reverse=True
        )
        betas = np.diff(curve, prepend=.0)
        betas /= betas[0]
        number_of_features = sum(betas > threshold)
        return ordered_features[:number_of_features]


@dataclass
class Selection:
    preselection: List[str]
    mis: pd.Series
    hsic_selection: List[str]
    mi_ordered_features: List[str]
    hsic_ordered_features: List[str]
    lassopaths: pd.DataFrame
    regcurve: np.ndarray
    features: List[str] = None

    def select_from_lasso_path(self, threshold: float = 0.01):
        return HSICSelector.select_from_lasso_path(self.lassopaths, threshold)


def select(
    x: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    mi_threshold: float = .00001,
    hsic_threshold: float = .01,
    batch_size=9000,
    minibatch_size: int = 800,
    number_of_epochs: int = 3,
    use_preselection: bool = False,
    device: Device = Device.CPU,
) -> Selection:
    n, d = x.shape
    if use_preselection:
        cols, mis = ksgmi(x, y, mi_threshold)
    else:
        cols = x.columns.tolist()
        mis = pd.Series(np.zeros(d), index=cols)
    x_ = x.loc[:, cols].values
    y_ = y.values
    if y_.ndim == 1:
        y_ = y_.reshape(-1, 1)
    selector = HSICSelector(x_, y_, feature_names=cols)
    innersel_ = selector.autoselect(
        threshold=hsic_threshold,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        number_of_epochs=number_of_epochs,
        device=device
    )
    print(f'HSIC has selected {len(innersel_)} features')
    hsic_ordered_features = list(
        np.array(cols)[selector.ordered_features]
    )
    preselection: List[str] = cols
    mi_ordered_features: List[str] = list(
        mis.sort_values(ascending=False).index)
    hsic_selection: List[str] = innersel_
    paths: pd.DataFrame = selector.lasso_path()
    curve: np.array = np.cumsum(np.sort(paths.iloc[-1, :])[::-1])
    features: List[str] = hsic_selection
    sel = Selection(
        preselection=preselection,
        mis=mis,
        hsic_selection=hsic_selection,
        mi_ordered_features=mi_ordered_features,
        hsic_ordered_features=hsic_ordered_features,
        lassopaths=paths,
        regcurve=curve,
        features=features,
    )
    return sel


def _make_batches(x, batch_size):
    n, _ = x.shape
    b = min(n, batch_size)
    num_batches = n // b
    batches = np.split(x[:num_batches * b, :], num_batches, axis=0)
    return batches


def preprocess(
        x: np.ndarray,
        xfeattype: FeatureType,
        y: np.ndarray,
        yfeattype: FeatureType,
        repeat: int = 1,
        standard: bool = False,
        catcont_split: Optional[int] = None,
):
    assert x.ndim == 2
    assert y.ndim == 2
    n, d = x.shape
    assert y.shape[0] == n
    if xfeattype == FeatureType.DISCR:
        if x.dtype not in (np.int32, np.int64):
            raise ValueError(x.dtype)
        catcont_split = d
    elif xfeattype == FeatureType.CONT:
        catcont_split = 0
        x = x.astype(float)
    elif xfeattype == FeatureType.BOTH:
        catcont_split = int(catcont_split) if catcont_split else 0
        assert 0 <= catcont_split and catcont_split <= d
    if yfeattype == FeatureType.DISCR:
        if y.dtype not in (np.int32, np.int64):
            raise ValueError(y.dtype)
    elif yfeattype == FeatureType.CONT:
        y = y.astype(float)
    if standard:
        if xfeattype != FeatureType.DISCR:
            x[:, catcont_split:] = (x[:, catcont_split:] - np.sum(x[:, catcont_split:], axis=0, keepdims=True)) / \
                (1e-9 + np.std(x[:, catcont_split:], axis=0, keepdims=True))
        if yfeattype != FeatureType.DISCR:
            y = (y - np.sum(y, axis=0, keepdims=True)) / \
                (1e-9 + np.std(y, axis=0, keepdims=True))
    idxs: List[np.ndarray] = [np.random.permutation(n) for _ in range(repeat)]
    xs: List[np.ndarray] = [x[idx, :] for idx in idxs]
    ys: List[np.ndarray] = [y[idx, :] for idx in idxs]
    xx = np.vstack(xs)
    yy = np.vstack(ys)
    return xx, yy


def _to_projection_matrix(features: List[int], d: int, f: int) -> np.ndarray:
    z = len(features)
    p = np.zeros(shape=(f, d), dtype=float)
    for i, j in zip(range(z), features):
        p[i, j] = 1.
    return p


def _to_feature_list(p: np.ndarray) -> List[int]:
    assert p.ndim == 2
    f, d = p.shape
    ranking = np.argsort(
        np.sum(np.abs(p), axis=0)
    )[::-1]
    features = list(ranking[:f])
    return features


def _run(
        x: np.ndarray,
        y: np.ndarray,
        number_of_features: int,
        batch_size: int = 500,
        xkerneltype: Optional[KernelType] = None,
        ykerneltype: Optional[KernelType] = None,
        catcont_split: Optional[int] = None,
        device: Device = Device.CPU,
):
    if xkerneltype is None:
        xkerneltype = KernelType.DELTA if x.dtype in (
            np.int32, np.int64) else KernelType.RBF
    if ykerneltype is None:
        ykerneltype = KernelType.DELTA if y.dtype in (
            np.int32, np.int64) else KernelType.RBF
    nx, dx = x.shape
    ny, dy = y.shape
    assert nx == ny
    num_batches: int = nx // batch_size
    assert num_batches >= 1
    gram_dim: int = num_batches * batch_size**2
    lx = 1.
    ly = np.sqrt(dy)
    x_gram: np.ndarray
    y_gram: np.ndarray

    gram_maker: Callable[[...], np.ndarray]
    if device == Device.GPU:
        gram_maker = cudakernels.apply_feature_map
    else:
        gram_maker = kernels.apply_feature_map

    x_gram = gram_maker(
        xkerneltype,
        x.T, lx,
        batch_size,
        is_multivariate=False,
        catcont_split=catcont_split,
        device=device)
    y_gram = gram_maker(
        ykerneltype,
        y.T, ly,
        batch_size,
        is_multivariate=True,
        device=device)
    assert x_gram.shape == (gram_dim, dx)
    assert not np.any(np.isnan(x_gram))
    assert y_gram.shape == (gram_dim, 1)
    assert not np.any(np.isnan(y_gram))
    features, lassopath = lar.solve(x_gram, y_gram, number_of_features)
    return features, lassopath
