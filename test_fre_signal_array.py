# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:38:00 2023

@author: Baarbod
"""

# test fre signal equation. it should give the same result as the simpler 
# expression that assumes all relaxation times are TR. 

from fre_signal import fre_signal
from fre_signal import fre_signal_array as fre_signal2
import numpy as np
import time

fa = 47*np.pi/180
TR = 0.35
T1 = 4
#dt_list = np.array([float('nan'), 0.2, 0.3, 0.23, 0.4, 0.323, 0.28, 0.16, 0.62, 0.2, 0.3, 0.23, 0.4, 0.323, 0.28, 0.16, 0.62, 0.2, 0.3, 0.23, 0.4, 0.323, 0.28, 0.16, 0.62])
dt_list = np.array([float('nan'), TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR])
#dt_list = np.array([float('nan')])
n = len(dt_list)
M0 = 1

t = time.time()
for i in range(100):
    # test typical case
    S1 = fre_signal(n, fa, TR, T1, dt_list)
elapsed = time.time() - t
print(elapsed)

t = time.time()
for i in range(100):
    S2 = fre_signal2(n, fa, TR, T1, dt_list)
elapsed = time.time() - t
print(elapsed)


