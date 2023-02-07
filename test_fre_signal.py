# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:38:00 2023

@author: Baarbod
"""

# test fre signal equation. it should give the same result as the simpler 
# expression that assumes all relaxation times are TR. 

from fre_signal import fre_signal
from fre_signal_array import fre_signal as fre_signal2
import numpy as np
import matplotlib.pyplot as plt


#n = 9
n = 6
fa = 47*np.pi/180
TR = 0.35
T1 = 4
#dt_list = np.array([float('nan'), 0.2, 0.3, 0.23, 0.4, 0.323, 0.28, 0.16, 0.62])
dt_list = np.array([float('nan'), TR, TR, TR, TR, TR])
M0 = 1


# test typical case
#S = fre_signal(n, fa, TR, T1, dt_list)
S = fre_signal2(n, fa, TR, T1, dt_list)

# test against other formulation 
S1 = np.zeros(n)
S2 = np.zeros(n)
for ipulse in range(n):
    S1[ipulse] = fre_signal(ipulse+1, fa, TR, T1, dt_list)
    S2[ipulse] = M0*np.sin(fa)*(np.exp(-TR/T1)*np.cos(fa))**(ipulse+1-1) \
        *(1 - (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(fa)))
        
# Plot slice signals
plt.plot(S1)
plt.show()

# Plot slice signals
plt.plot(S2)
plt.show()


