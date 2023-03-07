# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:40:07 2023

@author: Baarbod
"""

import numpy as np
import matplotlib.pyplot as plt
from tof.posfunclib import compute_position_fourier
from tof.posfunclib import compute_position_fourier_spatial
from tof.posfunclib import compute_position_sine_spatial
from tof.posfunclib import compute_position_sine
from functools import partial 
plt.style.use('seaborn-poster')

tmax = 100
t_eval = np.arange(0, tmax, 0.1)
# x = compute_position_sine_spatial(t_eval, 0, -1, 1, 2*np.pi/5)

main_frequencies = [0.05, 0.2, 0.9]
# define position function for model
An = np.array([0, 0.3, 0.1, 0.5])
Bn = np.array([0, 0, 0])
w0 = 2*np.pi*np.array(main_frequencies)
Xfunc = partial(compute_position_fourier_spatial, An=An, Bn=Bn, w0=w0)
x = Xfunc(t_eval, 0)

plt.figure(figsize = (12, 4))
plt.subplot(121)
plt.plot(t_eval, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.subplot(122)
v = np.diff(x)/np.mean(np.diff(t_eval))
plt.plot(t_eval[1:], v)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.tight_layout()
plt.show()

#x2 = compute_position_sine(t_eval, 0, -1, 1, 2*np.pi/5)

Xfunc2 = partial(compute_position_fourier, An=An, Bn=Bn, w0=w0)
x2 = Xfunc2(t_eval, 0)

plt.figure(figsize = (12, 4))
plt.subplot(121)
plt.plot(t_eval, x2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.subplot(122)
v = np.diff(x2)/np.mean(np.diff(t_eval))
plt.plot(t_eval[1:], v)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.tight_layout()
plt.show()