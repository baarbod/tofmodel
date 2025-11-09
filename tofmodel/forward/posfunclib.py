# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def compute_position_numeric_spatial(t_eval, x0, tr_vect, vts, xarea, area, solver_method='RK23'):
    """ Perform numerical integration of v(x, t) to obtain positions at each time point
    
    Parameters
    ----------
    t_eval : numpy.ndarray
        time points (s) to evaluate position (cm)
        
    x0 : numpy.ndarray
        initial positions (cm)
        
    tr_vect : numpy.ndarray_
        time points (s) associated with numerical input velocity (cm/s)
        
    vts : numpy.ndarray
        velocity timeseries (cm/s)
        
    xarea : numpy.ndarray
        position vector (cm) associated with cross-sectional areas (cm^2)
        
    area : numpy.ndarray
        cross-sectional areas (cm)

    Returns
    -------
    sol.y : numpy.ndarray
        positions (cm) at each time point
    """
    
    if np.ndim(x0) == 0:
        x0 = np.expand_dims(x0, axis=0)
    
    ind0 = np.abs(xarea).argmin() # find where xarea is zero
    area0 = area[ind0]
    v_interp = interp1d(tr_vect, vts, kind='linear', bounds_error=False, fill_value='extrapolate')
    area_interp = interp1d(xarea, area, kind='linear', bounds_error=False, fill_value=(area[0], area[-1]))

    def func(t, x):
        a = area_interp(x)
        v = v_interp(t)
        return (area0 / a) * v

    trange = [np.min(t_eval), np.max(t_eval)]
    sol = solve_ivp(func, trange, x0, t_eval=t_eval, method=solver_method, vectorized=True)

    return sol.y


def compute_position_constant(t, x0, v0):
    x0 = x0[:, np.newaxis]  # Now x0 has shape (n, 1)
    t = t[np.newaxis, :]    # Now t has shape (1, m)
    X = v0 * t + x0
    return X


def compute_position_sine(t, x0, v1, v2, w0):
    x0 = x0[:, np.newaxis]  # Now x0 has shape (n, 1)
    t = t[np.newaxis, :]    # Now t has shape (1, m)
    amplitude = (v2 - v1) / 2
    offset = v1 + amplitude
    return offset*t + amplitude / w0 * np.sin(w0 * t) + x0


def compute_position_fourier(t, x0, an, bn, w0):
    offset = np.array(an[0])
    an = np.array(an[1:])
    bn = np.array(bn)
    n = np.size(an)
    n_vector = np.ones((n, 1))

    an = np.reshape(an, (n, 1))
    bn = np.reshape(bn, (n, 1))
    w0 = np.reshape(w0, (n, 1))
    k = np.sum(bn / (w0 * n_vector)) + x0
    if n == 1:
        term = an / w0 * np.sin(w0 * t) - bn / w0 * np.cos(w0 * t)
    else:
        tt = an / (w0 * n_vector) * np.sin(w0 * n_vector * t) - bn / (w0 * n_vector) * np.cos(w0 * n_vector * t)
        term = np.sum(tt, axis=0)
    term = term.squeeze()
    return offset * t / 2 + term + k


def compute_position_fourier_phase(t, x0, an, bn, w0, phase):
    offset = np.array(an[0])
    an = np.array(an[1:])
    bn = np.array(bn)
    n = np.size(an)
    n_vector = np.ones((n, 1))

    an = np.reshape(an, (n, 1))
    bn = np.reshape(bn, (n, 1))
    w0 = np.reshape(w0, (n, 1))
    phase = np.reshape(phase, (n, 1))
    k = np.sum(bn / (w0 * n_vector)) + x0
    if n == 1:
        term = an / w0 * np.sin(w0 * (t-phase)) - bn / w0 * np.cos(w0 * (t-phase))
    else:
        tt = an / (w0 * n_vector) * np.sin(w0 * n_vector * (t-phase)) - bn / (w0 * n_vector) * np.cos(w0 * n_vector * (t-phase))
        term = np.sum(tt, axis=0)
    term = term.squeeze()
    return offset * t / 2 + term + k


