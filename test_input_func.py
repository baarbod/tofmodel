# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:24:53 2023

@author: Baarbod
"""

import numpy as np
from functools import partial 
import matplotlib.pyplot as plt

def model(Xfunc, t):
    x = Xfunc(t, 0)
    return x

def compute_position_constant(t, x0, v0):
    X = v0*t + x0
    return X

def compute_position_sine(t, x0, v0):
    v1 = 0
    v2 = v0
    Amp = (v2-v1)/2
    A0 = (v1 + Amp)*2
    An = Amp
    w0 = 2*np.pi/5
    X = A0*t/2 + An/w0*np.sin(w0*t) + x0 
    return X

def compute_position_fourier(t, x0, An, Bn, w0):
    
# FIX THIS
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
    return A0*t/2 + term + k


Xfunc = partial(compute_position_constant, v0=0.5)
#Xfunc = partial(compute_position_sine, v0=0.5)

# Run model using X function
t = np.arange(0, 100, 0.1)
x = model(Xfunc, t)

# Plot slice signals
plt.plot(t, x)
plt.show()


# x0 = 0
# An = [0, 0.2, 0.3]
# Bn = [0.1, 0.4]
# w0 = 2*np.pi/5
# X = compute_position_fourier(t, x0, An, Bn, w0)



