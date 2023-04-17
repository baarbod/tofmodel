# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:47:25 2023

@author: Baarbod
"""

import numpy as np
from tof import tofmodel as tm
import matplotlib.pyplot as plt
import tof.posfunclib as pfl
from functools import partial 
from scipy.integrate import cumtrapz
import inflowan.utils as utils

plt.close('all')

npulse = 50
tr = 1
trvect = tr * np.arange(npulse)
v1 = -0.6
v2 = 0.6
w0 = 2*np.pi/6
Amp = (v2-v1)/2
A0 = (v1 + Amp)*2
An = Amp
vin = A0/2 + An*np.cos(w0*trvect) 

vin = np.expand_dims(vin, axis=1)
vin = utils.upsample(vin, 5, tr)
tr = tr/5
trvect = tr * np.arange(np.size(vin))

A = np.loadtxt('area.txt')
fig, ax1 = plt.subplots(nrows=1, ncols=1)
slc1 = 30
xarea = 1 * np.arange(np.size(A)) - slc1
ax1.plot(xarea, A)

xcs = cumtrapz(np.squeeze(vin), trvect, initial=0)
Xfunc_num = partial(pfl.compute_position_numeric, trvect=trvect, xcs=xcs)
Xfunc_num_spa = partial(pfl.compute_position_numeric_spatial, trvect=trvect, vts=vin, xarea=xarea, A=A)

Xfunc_sine = partial(pfl.compute_position_sine, v1=v1, v2=v2, w0=w0)
Xfunc_sine_spa = partial(pfl.compute_position_sine_spatial, v1=v1, v2=v2, w0=w0, xarea=xarea, A=A)

# Dictionary containing model parameters
scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.504,
    'flip_angle' : 45,
    't1_time' : 4,
    'num_slice' : 21,
    'num_pulse' : npulse,
    'MBF' : 3, 
    'alpha_list' : [0,0.2775,0.07,0.345,0.1375,0.415,0.2075, 0,0.2775,0.07,0.345,0.1375,0.415,0.2075, 0,0.2775,0.07,0.345,0.1375,0.415,0.2075]}

S_num = tm.run_tof_model(scan_param, Xfunc_num)
S_num_spa = tm.run_tof_model(scan_param, Xfunc_num_spa)
S_sine = tm.run_tof_model(scan_param, Xfunc_sine)
S_sine_spa = tm.run_tof_model(scan_param, Xfunc_sine_spa)

trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])

# plot filtered signals
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].plot(trvect_sim, S_num[:, 0:3])
axes[1, 0].plot(trvect_sim, S_num_spa[:, 0:3])
axes[0, 1].plot(trvect_sim, S_sine[:, 0:3])
axes[1, 1].plot(trvect_sim, S_sine_spa[:, 0:3])

plt.show()
