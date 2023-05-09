import itertools
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


def first_index(
        subset: Set[int],
        permutation: List[int]
):
    m = 1
    for k in range(len(permutation)):
        if subset.issubset(set(permutation[:k+1])):
            return m - len(subset)
        m += 1
    return m - len(subset)


def global_first_index(
        subset: Set[int],
        permutations: List[List[int]],
        d: int,
):
    m = d
    for permutation in permutations:
        m_ = first_index(subset, permutation)
        if m_ < m:
            m = m_
    return m


def rho(
        permutations: List[List[int]],
        d: int,
):
    normalisation = 2**d - 1
    m = 0.
    for subset_size in range(1, d+1):
        for subset in itertools.combinations(range(d), subset_size):
            m += global_first_index(set(subset), permutations, d)
    return m / normalisation


def all_rhos(
    num_permutations: int,
    d: int
):
    all_ = {permutations: rho(permutations, d)
            for permutations in itertools.combinations(
                itertools.permutations(range(d)), num_permutations)
            }
    return all_


def rho_range(
    num_permutations: int,
    d: int
):
    max_ = 0.
    min_ = float(d+1)
    for permutations in itertools.combinations(
            itertools.permutations(range(d)), num_permutations):
        r = rho(permutations, d)
        if r < min_:
            min_ = r
        if max_ < r:
            max_ = r
    return min_, max_
