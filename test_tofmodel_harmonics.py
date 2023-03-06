# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:13:57 2023

@author: Baarbod
"""

import numpy as np
from tofmodel import run_tof_model
import matplotlib.pyplot as plt
import posfunclib as pfl
from functools import partial 

main_frequencies = [0.05, 0.2, 1]

# run tof model

A_resp = np.array([0, 0.4, 0.1, 0.1])

An = np.array([0, 0.4, 0.1, 0.1])
Bn = np.array([0, 0, 0])
w0 = 2*np.pi*np.array(main_frequencies)
Xfunc1 = partial(pfl.compute_position_fourier, An=An, Bn=Bn, w0=w0)
Xfunc2 = partial(pfl.compute_position_fourier_spatial, An=An, Bn=Bn, w0=w0)

scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.387,
    'flip_angle' : 47,
    't1_time' : 4,
    'num_slice' : 10,
    'num_pulse' : 200,
    'MBF' : 2, 
    'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775, 0.14, 0, 0.2075, 0.07, 0.2775]}

trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])
signal1 = run_tof_model(scan_param, Xfunc1)
signal2 = run_tof_model(scan_param, Xfunc2)

# plot filtered signals
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(trvect_sim, signal1[:, 0:4])
ax2.plot(trvect_sim, signal2[:, 0:4])
plt.show()
