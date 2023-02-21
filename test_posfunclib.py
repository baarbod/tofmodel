# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:54:59 2023

@author: Baarbod
"""

import numpy as np
from functools import partial 
import matplotlib.pyplot as plt
import posfunclib as pfl


t = np.arange(0, 100, 0.01)


Xfunc1 = partial(pfl.compute_position_constant, v0=0.1)
Xfunc2 = partial(pfl.compute_position_sine, v1=0, v2=0.2, w0=2*np.pi/5)

#An = [0, 0.2, 0.3, 0.6, 0.8, 0.4, 0.2, 0.3, 0.6, 0.8, 0.4]
#Bn = [0.1, 0.4, 0.9, 0.2, 0.1, 0.2, 0.3, 0.6, 0.8, 0.4]
An = [0, 0.1, 0.1, 0.1]
Bn = [0, 0, 0]
w0 = 2*np.pi*np.array([1/5, 1/2, 1])
Xfunc3 = partial(pfl.compute_position_fourier, An=An, Bn=Bn, w0=w0)

x1 = pfl.model(Xfunc1, t)
x2 = pfl.model(Xfunc2, t)
x3 = pfl.model(Xfunc3, t)

# Plot slice signals
plt.plot(t, x1)
plt.show()
plt.plot(t, x2)
plt.show()
plt.plot(t, x3)
plt.show()