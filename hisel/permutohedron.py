from typing import Optional, Set, Tuple, List
import numpy as np
from scipy.stats import special_ortho_group


def haar_sampling(
        d: int,
        size: int = 1,
        random_state: Optional[int] = None,
) -> Set[Tuple[int, ...]]:
    def projection(d: int):
        p = np.diag(np.arange(-1, -d, -1, dtype=float), 1)
        p += np.eye(d)
        for k in range(1, d):
            p += np.diag(np.ones(shape=d-k), -k)
        p = p[:d-1, :]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p
    u = special_ortho_group.rvs(d-1, size=size, random_state=random_state)
    if u.ndim == 2:
        u = np.expand_dims(u, axis=0)
    xs = np.concatenate((u, -u), axis=2)
    p = np.expand_dims(projection(d).T, axis=0)
    perms = np.argsort(p @ xs, axis=1)
    permutations = set([
        tuple(sigma) for perm in perms for sigma in perm.T])
    return permutations
