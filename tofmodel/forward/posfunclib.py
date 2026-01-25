import numpy as np
from scipy.integrate import solve_ivp


def compute_position_numeric_spatial(t_eval, x0, tr_vect, vts, xarea, area, solver_method='RK23'):
    if np.ndim(x0) == 0:
        x0 = np.atleast_1d(x0)
    ind0 = np.abs(xarea).argmin()
    area0 = area[ind0]
    def func(t, x):
        a = np.interp(x, xarea, area, left=area[0], right=area[-1])
        v = np.interp(t, tr_vect, vts)
        return (area0 / a) * v
    trange = (t_eval[0], t_eval[-1])
    sol = solve_ivp(func, trange, x0, t_eval=t_eval, method=solver_method, vectorized=True)
    return sol.y

def compute_position_constant(t, x0, v0):
    x0 = x0[:, np.newaxis]  # Now x0 has shape (n, 1)
    t = t[np.newaxis, :]    # Now t has shape (1, m)
    X = v0 * t + x0
    return X
