#  Least angle regression
import numpy as np
from scipy.sparse import lil_matrix


def solve(
        x: np.ndarray,
        y: np.ndarray,
        dim_z: int,
):
    """
    Solves the problem argmin_beta 1/2||y-X*beta||_2^2  s.t. beta>=0.
    The problem is solved using a modification of the Least Angle Regression
    and Selection algorithm.
    We used the a Python implementation of the Nonnegative LARS solver
    written in MATLAB at http://orbit.dtu.dk/files/5618980/imm5523.zip
    """
    assert dim_z > 0
    n, d = x.shape
    xx: np.ndarray = x.T @ x
    g: np.ndarray = x.T @ y
    beta: np.ndarray = np.zeros((d, 1), dtype=float)
    gb: np.ndarray = np.zeros((d, 1), dtype=float)
    c: np.ndarray = np.array(g, copy=True)
    gw: np.ndarray = np.zeros((d, 1), dtype=float)
    gamma: np.ndarray = np.zeros((d+1, 1), dtype=float)
    maxc: float
    j: int
    t: int
    aj: int
    ij: int

    active: List[int] = []
    indices: List[int] = list(range(d))
    num_active: int = 0
    lasso_cond: int

    path = lil_matrix((3*d, d))

    # first step
    k = 0  # index for path
    j = np.argmax(c)
    maxc = np.amax(c)  # c[j]
    active.append(j)
    indices.remove(j)
    num_active = 1

    # iteration
    while np.sum(c[active]) / num_active >= 1e-12 and num_active <= dim_z:
        w, _, _, _ = np.linalg.lstsq(
            x[:, active].T @  x[:, active],
            np.ones((num_active, 1), dtype=float),
            rcond=None
        )
        gw = x.T @ x[:, active] @ w
        denom = gw[active[0]] - gw[indices]
        if np.any(denom == 0.):
            print('ZeroDivisionError: Results of selection is unrelieable')
            print('Increase the batch size and try again')
            break
        gamma[:d - num_active] = (maxc - c[indices]) / denom

        gamma[d - num_active: d] = -beta[active] / w
        gamma[d] = c[active[0]] / gw[active[0]]

        gamma[gamma <= 1e-12] = np.inf
        t = np.argmin(gamma)
        beta[active] = beta[active] + gamma[t] * w

        if d - num_active <= t and t < d:
            lasso_cond = 1
            j = t - d + num_active
            aj = active[j]
            indices.append(aj)
            active.remove(aj)
        else:
            lasso_cond = 0

        gb = xx @ beta
        c = g - gb
        j = np.argmax(c[indices])
        maxc = np.amax(c[indices])  # c[indices][j]
        path[k, :] = beta
        if lasso_cond == 0:
            ij = indices[j]
            active.append(ij)
            indices.remove(ij)

        # update number of active variables
        num_active = len(active)
        k += 1

    if len(active) > dim_z:
        active.pop()

    lassopath = path[:k, :].toarray()

    del xx
    del gamma
    del g
    del beta

    return active, lassopath
