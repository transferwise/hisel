# API
from typing import List
import numpy as np
from hisel import lar, kernels


class Selector:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 2
        nx, dx = x.shape
        ny, dy = y.shape
        print('\nHSIC feature selection')
        print(f'Total number of features: {dx}')
        print(f'Dimensionality of target: {dy}')
        print(f'Number of x samples: {nx}')
        print(f'Number of y samples: {ny}')
        print('\n')
        assert nx == ny, 'number of samples in x and in y must be equal'
        self.x = np.array(x, copy=True)
        self.y = np.array(y, copy=True)

    def projection_matrix(self,
                          number_of_features: int,
                          batch_size: int = 1000,
                          minibatch_size: int = 200,
                          number_of_epochs: int = 1
                          ) -> np.ndarray:
        p: np.ndarray = np.zeros((number_of_features, self.x.shape[1]))
        x_, y_ = preprocess(self.x, self.y, 1, standard=True)
        xs = _make_batches(x_, batch_size)
        ys = _make_batches(y_, batch_size)
        for x, y in zip(xs, ys):
            x, y = preprocess(x, y, number_of_epochs, standard=False)
            features: List[int] = _run_numpy(
                x, y, number_of_features, minibatch_size)
            p += _to_projection_matrix(
                features,
                self.x.shape[1],
                number_of_features,
            )
        p /= len(xs)
        return p

    def select(self,
               number_of_features: int,
               batch_size: int = 1000,
               minibatch_size: int = 200,
               number_of_epochs: int = 1
               ) -> List[int]:
        p = self.projection_matrix(
            number_of_features=number_of_features,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            number_of_epochs=number_of_epochs,
        )
        features = _to_feature_list(p)
        return features


def _make_batches(x, batch_size):
    n, _ = x.shape
    b = min(n, batch_size)
    num_batches = n // b
    batches = np.split(x[:num_batches * b, :], num_batches, axis=0)
    return batches


def preprocess(
        x: np.ndarray,
        y: np.ndarray,
        repeat: int = 1,
        standard: bool = False,
):
    assert x.ndim == 2
    assert y.ndim == 2
    n, d = x.shape
    assert y.shape[0] == n
    if standard:
        x = (x - np.sum(x, axis=0, keepdims=True)) / \
            (1e-9 + np.std(x, axis=0, keepdims=True))
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


def _run_numpy(
        x: np.ndarray,
        y: np.ndarray,
        number_of_features: int,
        batch_size: int = 500
):
    nx, dx = x.shape
    ny, dy = y.shape
    assert nx == ny
    num_batches: int = nx // batch_size
    gram_dim: int = num_batches * batch_size**2
    lx = 1.
    ly = np.sqrt(dy)
    x_gram: np.ndarray = kernels.apply_feature_map(
        x.T, lx, batch_size, is_multivariate=False)
    assert x_gram.shape == (gram_dim, dx)
    assert not np.any(np.isnan(x_gram))
    y_gram: np.ndarray = kernels.apply_feature_map(
        y.T, ly, batch_size, is_multivariate=True)
    assert y_gram.shape == (gram_dim, 1)
    assert not np.any(np.isnan(y_gram))
    features = lar.solve(x_gram, y_gram, number_of_features)
    return features
