# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:38:00 2023

@author: Baarbod
"""

from fre_signal import fre_signal
import numpy as np


n = 9
#n = 1
fa = 47*np.pi/180
TR = 0.35
T1 = 4
dt_list = np.array([float('nan'), 0.2, 0.3, 0.23, 0.4, 0.323, 0.28, 0.16, 0.62])

S = fre_signal(n, fa, TR, T1, dt_list)