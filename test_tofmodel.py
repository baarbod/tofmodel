# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:02:15 2023

@author: Baarbod
"""

import numpy as np
from tofmodel import run_tof_model
import matplotlib.pyplot as plt
import posfunclib as pfl
from functools import partial 
#import input_handler as ih

# Position functions
#Xfunc = partial(pfl.compute_position_constant, v0=0.1)
#Xfunc = partial(pfl.compute_position_sine, v1=-0.5, v2=0.5, w0=2*np.pi/5)
#Xfunc = partial(pfl.compute_position_sine, v1=-0.5, v2=0.5, w0=2*np.pi/5)
Xfunc = partial(pfl.compute_position_sine_spatial, v1=-0.5, v2=0.5, w0=2*np.pi/5)

# An = [0.2, 0.1]
# Bn = [0*0.1]
# w0 = 2*np.pi/5
# Xfunc = partial(pfl.compute_position_fourier, An=An, Bn=Bn, w0=w0)

#scan_param = ih.input_from_json('test_scan.json')

# Dictionary containing model parameters
scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.35,
    'flip_angle' : 47,
    't1_time' : 4,
    'num_slice' : 10,
    'num_pulse' : 100,
    'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775, 0.14, 0, 0.2075, 0.07, 0.2775]}

signal = run_tof_model(scan_param, Xfunc)
trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])

# plot filtered signals
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(trvect_sim, signal[:, 0:4])
plt.show()
