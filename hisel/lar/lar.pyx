# distutils: language = c++
# cython: language_level = 3

# Least angle regression
import numpy as np
cimport numpy as np
from scipy.sparse import lil_matrix

DTYPEf = np.float64 
DTYPEi = np.int64
ctypedef np.float_t  DTYPEf_t
ctypedef np.int_t DTYPEi_t

def solve(
        np.ndarray[DTYPEf_t, ndim=2] x,
        np.ndarray[DTYPEf_t, ndim=2] y,
        int dim_z,
        ):
    """
    Solves the problem argmin_beta 1/2||y-X*beta||_2^2  s.t. beta>=0.
    The problem is solved using a modification of the Least Angle Regression
    and Selection algorithm.
    We used the a Python implementation of the Nonnegative LARS solver
    written in MATLAB at http://orbit.dtu.dk/files/5618980/imm5523.zip
    """

    assert dim_z>0
    cdef int n = x.shape[0]
    cdef int d = x.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=2] xx = x.T @ x
    cdef np.ndarray[DTYPEf_t, ndim=2] g = x.T @ y
    cdef np.ndarray[DTYPEf_t, ndim=2] beta = np.zeros((d, 1), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] gb = np.zeros((d, 1), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] c = np.array(g, copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=2] gw  = np.zeros((d, 1), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] gamma  = np.zeros((d+1, 1), dtype=DTYPEf)
    cdef float maxc
    cdef int j, t, aj, ij
    cdef list active = []
    cdef list indices = list(range(d))
    cdef int num_active = 0
    cdef int lasso_cond 

    path = lil_matrix((3*d, d))

    # first step
    k = 0 # index for path
    j = np.argmax(c)
    maxc = np.amax(c) # c[j] 
    active.append(j)
    indices.remove(j)
    num_active = 1

    # iteration
    while np.sum(c[active]) / num_active >= 1e-12 and num_active <= dim_z:
        w, _, _, _ = np.linalg.lstsq(
                x[:, active].T @  x[:, active] , 
                np.ones((num_active, 1), dtype=DTYPEf),
                rcond=None
                )
        gw = x.T @ x[:, active] @ w
        gamma[:d - num_active] = (maxc - c[indices]) / (gw[active[0]] - gw[indices])

        gamma[d - num_active: d] = -beta[active] / w
        gamma[d] = c[active[0]] / gw[active[0]]

        gamma[gamma<=1e-12] = np.inf
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
        k+=1

    if len(active) > dim_z:
        active.pop()


    lassopath = path[:k, :].toarray()

    return active, lassopath




