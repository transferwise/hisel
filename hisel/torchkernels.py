from typing import Optional
import torch
from torch import Tensor


def featwise(
        x: Tensor,
        l: float
) -> Tensor:
    assert x.ndim == 2
    d, n = x.size()
    z = x.reshape(d, n, 1)
    s = x.reshape(d, 1, n)
    s2 = torch.square(s).expand(-1, n, -1)
    z2 = torch.transpose(s2, 1, 2)
    delta = z2 + s2 - 2*torch.bmm(z,  s)
    grams = torch.exp(-delta / (2*l*l))
    return grams


def multivariate(
        x: Tensor,
        y: Tensor,
        l: float
) -> Tensor:
    nx = x.size(1)
    ny = y.size(1)
    x2 = torch.sum(
        torch.square(x), dim=0, keepdims=True).expand(ny, -1)
    y2 = torch.sum(
        torch.square(y), dim=0, keepdims=True).expand(nx, -1)
    delta = x2.T + y2 - 2 * x.T @ y
    gram = torch.exp(-delta / (2 * l * l))
    return gram


def multivariate_phi(
        x: Tensor,
        l: float
) -> Tensor:
    gram = multivariate(x, x, l)
    n, m = gram.size()
    gram = gram.reshape(1, n, m)
    return gram


def _centering_matrix(d: int, n: int, dtype=torch.float64) -> Tensor:
    id_ = torch.eye(n, dtype=dtype)
    ids = id_.reshape(1, n, n).expand(d, -1, -1)
    ones = torch.ones_like(ids)
    h = ids - ones / n
    return h


def _center_gram(
        g: Tensor,
        h: Optional[Tensor] = None
) -> Tensor:
    if h is None:
        h = _centering_matrix(g.size(0), g.size(2))
    return torch.bmm(h, torch.bmm(g,  h))


def _run_batch(
        x: Tensor,
        l: float,
        h: Optional[Tensor] = None,
        is_multivariate: bool = False,
) -> Tensor:
    phi = multivariate_phi if is_multivariate else featwise
    grams: Tensor = _center_gram(phi(x, l), h)
    d, n, m = grams.size()
    assert n == m
    g: Tensor = grams.reshape(d, n*m).T
    return g


def _make_batches(x, batch_size):
    d, n = x.size()
    b = min(n, batch_size)
    num_batches = n // b
    batches = torch.split(x[:, :num_batches * b], b, dim=1)
    return batches


def apply_feature_map(
        x: Tensor,
        l: float,
        batch_size: int,
        is_multivariate: bool = False,
        no_parallel: bool = True  # Only kept to uniform notation with numpy kernels
) -> Tensor:
    d, n = x.size()
    b = min(n, batch_size)
    batches = _make_batches(x, batch_size)
    h: Tensor
    if not is_multivariate:
        h = _centering_matrix(d, b, x.dtype)
    else:
        h = _centering_matrix(1, b, x.dtype)
    h = h.to(x.device)
    phi: Tensor = torch.vstack(
        [_run_batch(
            batch,
            l,
            h,
            is_multivariate
        ) for batch in batches]
    )
    return phi
