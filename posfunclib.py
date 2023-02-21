# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:24:53 2023

@author: Baarbod
"""

import numpy as np
from scipy.integrate import solve_ivp


def model(Xfunc, t):
    x = Xfunc(t, 0)
    return x

def compute_position_constant(t, x0, v0):
    X = v0*t + x0
    return X

def compute_position_sine(t, x0, v1, v2, w0):
    Amp = (v2-v1)/2
    A0 = (v1 + Amp)*2
    An = Amp
    return A0*t/2 + An/w0*np.sin(w0*t) + x0 

def compute_position_sine_spatial(t_eval, x0, v1, v2, w0):

    def F(t, x, k, m, r1, v1, v2, w0):
        Amp = (v2-v1)/2
        A0 = (v1 + Amp)*2
        An = Amp
        pos_term = k*(r1/(m*x + r1))**4
        time_term = A0/2 + An*np.cos(w0*t) 
        state = pos_term * time_term 
        return state
    
    k = 1
    m = 0*0.5
    r1 = 1
    p = (k, m, r1, v1, v2, w0)
    
    trange = [np.min(t_eval), np.max(t_eval)]
    sol = solve_ivp(F, trange, [x0], args=p, t_eval=t_eval)
    return  sol.y[0]
    
def compute_position_fourier(t, x0, An, Bn, w0):
    
    A0 = np.array(An[0])
    An = np.array(An[1:])
    Bn = np.array(Bn)
    N = np.size(An)
    Nvect = np.arange(1, N+1, 1)
    
    An = np.reshape(An, (N, 1))
    Bn = np.reshape(Bn, (N, 1))
    Nvect = np.reshape(Nvect, (N, 1))
    k = np.sum(Bn/(w0*Nvect)) + x0
    if N == 1:
        term = An/w0*np.sin(w0*t) - Bn/w0*np.cos(w0*t)
    else:
        tt = An/(w0*Nvect)*np.sin(w0*Nvect*t) - Bn/(w0*Nvect)*np.cos(w0*Nvect*t)
        term = np.sum(tt, axis=0)
    term = term.squeeze()
    return A0*t/2 + term + k






