from typing import Optional
import numpy as np


def featwise(
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


def multivariate(
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
        x: np.ndarray,
        l: float,
        h: Optional[np.ndarray] = None
) -> np.ndarray:
    grams: np.ndarray = _center_gram(featwise(x, l))
    d, n, m = grams.shape
    assert n == m
    g: np.ndarray = np.reshape(grams, (d, n*m)).T
    return g


def apply_feature_map(
        x: np.ndarray,
        l: float,
        batch_size: int,
) -> np.ndarray:
    d, n = x.shape
    b = min(n, batch_size)
    num_batches = n // b
    batches = np.split(x[:, :num_batches * b], num_batches, axis=1)
    h = _centering_matrix(d, b)
    phi: np.ndarray = np.vstack(
        [_run_batch(
            batch,
            l,
            h
        ) for batch in batches]
    )
    return phi
