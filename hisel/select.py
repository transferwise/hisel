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
        if dy > 1:
            raise NotImplementedError(
                'mutlidimensional target has not been implemented yet')
        self.x = np.array(x, copy=True)
        self.y = np.array(y, copy=True)

    def select(self,
               number_of_features: int,
               batch_size: int = 500,
               data_augmentation: int = 0
               ) -> List[int]:
        x_, y_ = preprocess(self.x, self.y, 1 + data_augmentation)
        features: List[int] = _run_numpy(
            x_, y_, number_of_features, batch_size)
        return features

    def projection_matrix(self,  number_of_features: int, batch_size: int = 500, data_augmentation: int = 0):
        features = self.select(
            number_of_features, batch_size, data_augmentation)
        p = _to_projection_matrix(features, self.x.shape[1])
        return p


def preprocess(
        x: np.ndarray,
        y: np.ndarray,
        repeat: int = 1,
):
    assert x.ndim == 2
    assert y.ndim == 2
    n, d = x.shape
    assert y.shape[0] == n
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


def _to_projection_matrix(features: List[int], d: int) -> np.ndarray:
    z = len(features)
    p = np.zeros(shape=(z, d), dtype=float)
    for i, j in zip(range(z), features):
        p[i, j] = 1.
    return p


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
    lx = np.sqrt(dx)
    ly = np.sqrt(dy)
    x_gram: np.ndarray = kernels.apply_feature_map(
        x.T, lx, batch_size)
    assert x_gram.shape == (gram_dim, dx)
    y_gram: np.ndarray = kernels.apply_feature_map(
        y.T, ly, batch_size)
    assert y_gram.shape == (gram_dim, dy)
    features = lar.solve(x_gram, y_gram, number_of_features)
    return features
