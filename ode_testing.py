# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:56:03 2023

@author: bashe
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use('seaborn-poster')

def F(t, x, k, m, r1):
    return k*(r1/(m*x + r1))**2 * np.cos(t*2*np.pi/5)

k = 2*0.5
m = 0*0.5
r1 = 1
p = (k, m, r1)

t_eval = np.arange(0, 30, 0.1)
sol = solve_ivp(F, [0, 30], [0], args=p, t_eval=t_eval)

plt.figure(figsize = (12, 4))
plt.subplot(121)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('x(t)')
plt.subplot(122)
v = np.diff(sol.y[0])/np.mean(np.diff(t_eval))
plt.plot(sol.t[1:], v)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.tight_layout()
plt.show()