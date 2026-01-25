import numpy as np


def compute_position_numeric_spatial(t_eval, x0, tr_vect, vts, xarea, area):
    if np.ndim(x0) == 0:
        x0 = np.atleast_1d(x0)
    ind0 = np.abs(xarea).argmin()
    area0 = area[ind0]
    n_protons = len(x0)
    n_times = len(t_eval)
    X = np.zeros((n_protons, n_times))
    X[:, 0] = x0
    def get_dx_dt(t, x):
        a = np.interp(x, xarea, area, left=area[0], right=area[-1])
        v = np.interp(t, tr_vect, vts)
        return (area0 / a) * v
    for i in range(n_times - 1):
        t = t_eval[i]
        dt = t_eval[i+1] - t
        curr_x = X[:, i]
        k1 = get_dx_dt(t, curr_x)
        k2 = get_dx_dt(t + dt/2, curr_x + dt/2 * k1)
        k3 = get_dx_dt(t + dt/2, curr_x + dt/2 * k2)
        k4 = get_dx_dt(t + dt, curr_x + dt * k3)
        X[:, i+1] = curr_x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return X

def compute_position_constant(t, x0, v0):
    x0 = x0[:, np.newaxis]  # Now x0 has shape (n, 1)
    t = t[np.newaxis, :]    # Now t has shape (1, m)
    X = v0 * t + x0
    return X
