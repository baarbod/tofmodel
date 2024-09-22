# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp


def model(Xfunc, t):
    x = Xfunc(t, 0)
    return x


def compute_position_constant(t, x0, v0):
    X = v0 * t + x0
    return X


def compute_position_sine(t, x0, v1, v2, w0):
    amplitude = (v2 - v1) / 2
    offset = v1 + amplitude
    return offset*t + amplitude / w0 * np.sin(w0 * t) + x0


def compute_position_sine_phase(t, x0, v1, v2, w0, phase):
    amplitude = (v2 - v1) / 2
    offset = v1 + amplitude
    return offset*t + amplitude / w0 * np.sin(w0 * (t-phase)) + x0 - 0*amplitude / w0 * np.sin(-w0*phase)


def compute_position_sine_spatial(t_eval, x0, v1, v2, w0, xarea, area):
    def func(t, x, v1, v2, w0, xarea, area):
        # ind0 = xarea == 0
        ind0 = np.abs(xarea).argmin() # find where xarea is zero
        area0 = area[ind0]
        diffarray = np.absolute(xarea - x)
        ind = diffarray.argmin()
        a = area[ind]
        if a < 0.05:
            a = 0.05
        pos_term = area0 / a

        amplitude = (v2 - v1) / 2
        offset = (v1 + amplitude) * 2
        time_term = offset / 2 + amplitude * np.cos(w0 * t)
        state = pos_term * time_term
        return state

    p = (v1, v2, w0, xarea, area)

    trange = [np.min(t_eval), np.max(t_eval)]
    sol = solve_ivp(func, trange, [x0], args=p, t_eval=t_eval)
    return sol.y[0]


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


def compute_position_fourier_spatial(t_eval, x0, an, bn, w0, xarea, area):
    def func(t, x, an, bn, w0, xarea, area):
        # ind0 = xarea == 0
        ind0 = np.abs(xarea).argmin() # find where xarea is zero
        area0 = area[ind0]
        diffarray = np.absolute(xarea - x)
        ind = diffarray.argmin()
        a = area[ind]
        if a < 0.05:
            a = 0.05
        pos_term = area0 / a

        offset = np.array(an[0])
        an = np.array(an[1:])
        bn = np.array(bn)
        n = np.size(an)
        n_vector = np.ones((n, 1))

        an = np.reshape(an, (n, 1))
        bn = np.reshape(bn, (n, 1))
        w0 = np.reshape(w0, (n, 1))
        kk = np.sum(bn / (w0 * n_vector))
        if n == 1:
            term = an * np.cos(w0 * t) + bn * np.sin(w0 * t)
        else:
            tt = an * np.cos(w0 * n_vector * t) + bn * np.sin(w0 * n_vector * t)
            term = np.sum(tt, axis=0)
        term = term.squeeze()

        time_term = offset + term + kk
        state = pos_term * time_term
        return state

    p = (an, bn, w0, xarea, area)

    trange = [np.min(t_eval), np.max(t_eval)]
    sol = solve_ivp(func, trange, [x0], args=p, t_eval=t_eval, method='Radau')
    return sol.y[0]


def compute_position_triangle(t_eval, x0, a, v0, t):
    time_mod = t_eval % t
    n = np.floor(t_eval / t)
    x = np.zeros(np.shape(time_mod))

    for idx, t in enumerate(time_mod):
        offset = n[idx] * (t * a + (v0 - a) * t)
        if t < t / 2:
            x[idx] = 2 * a / t * t ** 2 + t * (v0 - a) + offset

        elif t >= t / 2:
            x[idx] = t / 2 * (v0 - a) + a * t / 2 + (v0 - a) * (t - t / 2) + 2 * a * (t - t / 2) - 2 * a * (
                        t - t / 2) ** 2 / t + offset

    return x + x0


def compute_position_numeric(t_eval, x0, tr_vect, xcs):
    x = np.zeros(np.size(t_eval))
    xcs = np.array(xcs)
    xcs += x0
    for idx, timing in enumerate(t_eval):
        diffarray = np.absolute(tr_vect - timing)
        ind = diffarray.argmin()
        x[idx] = xcs[ind]
    return x


def compute_position_numeric_spatial(t_eval, x0, tr_vect, vts, xarea, area):
    
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
    sol = solve_ivp(func, trange, x0, args=p, t_eval=t_eval, vectorize=True)
    return sol.y
