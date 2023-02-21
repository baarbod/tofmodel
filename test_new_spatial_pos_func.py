# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:40:07 2023

@author: Baarbod
"""

import numpy as np
import matplotlib.pyplot as plt
from posfunclib import compute_position_sine_spatial
from posfunclib import compute_position_sine

plt.style.use('seaborn-poster')

tmax = 100
t_eval = np.arange(0, tmax, 0.1)
x = compute_position_sine_spatial(t_eval, 0, -1, 1, 2*np.pi/5)

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

x2 = compute_position_sine(t_eval, 0, -1, 1, 2*np.pi/5)
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