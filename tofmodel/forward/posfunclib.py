# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp


def compute_position_numeric_spatial(t_eval, x0, tr_vect, vts, xarea, area):
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
    
    ind0 = np.abs(xarea).argmin() # find where xarea is zero
    area0 = area[ind0]
    area_clipped = np.clip(area, 0.05, None)

    def func(t, x, vts, xarea, area_clipped):

        diffarray = np.abs(xarea - x[:, np.newaxis])
        ind = diffarray.argmin(axis=1)
        a = area_clipped[ind]
        pos_term = area0 / a
        
        diffarray = np.absolute(tr_vect - t)
        ind = diffarray.argmin()
        time_term = vts[ind]
        
        state = pos_term * time_term
        return state


    p = (vts, xarea, area_clipped)

    trange = [np.min(t_eval), np.max(t_eval)]
    sol = solve_ivp(func, trange, x0, args=p, t_eval=t_eval, method='RK23')

    return sol.y


def compute_position_constant(t, x0, v0):
    x0 = x0[:, np.newaxis]  # Now x0 has shape (n, 1)
    t = t[np.newaxis, :]    # Now t has shape (1, m)
    X = v0 * t + x0
    return X


def compute_position_sine(t, x0, v1, v2, w0):
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


