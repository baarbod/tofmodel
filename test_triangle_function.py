# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:56:49 2023

@author: Baarbod
"""

import numpy as np
import matplotlib.pyplot as plt
from posfunclib import compute_position_triangle
from posfunclib import compute_position_sine
from functools import partial 
from tofmodel import run_tof_model
plt.style.use('seaborn-poster')

import sys
sys.path.append('C:/Users/Baarbod/projectlocal/inflowanalysis')
from flowanlib import derive_velocity

# define parameters for triangle wave
tmax = 30
t_eval = np.arange(0, tmax, 0.01)
x0 = 0
A = 0.4
V0 = 0.1
T = 6

# define triangle position function
Xfunc_triangle = partial(compute_position_triangle, A=A, V0=V0, T=T)

# plot the position and velocity over a time period
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
x = Xfunc_triangle(t_eval, x0)
ax1.plot(t_eval, x)
v = derive_velocity(t_eval, Xfunc_triangle, 0)
ax2.plot(t_eval[1:], v)
plt.show()

# define model parameters
scan_param =	{
    'slice_width' : 0.25,
    'repetition_time' : 0.387,
    'flip_angle' : 47,
    't1_time' : 4,
    'num_slice' : 10,
    'num_pulse' : 200,
    'MBF' : 2, 
    'alpha_list' : [0.14, 0, 0.2075, 0.07, 0.2775, 0.14, 0, 0.2075, 0.07, 0.2775]}

# solve the model using triangular wave input
trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])
S = run_tof_model(scan_param, Xfunc_triangle)

# plot model output
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(trvect_sim, S[:, 0:3])
ax.set_title('With triangle input')
plt.show()

# define a position function using a sine function
Xfunc_sine = partial(compute_position_sine, v1=-1, v2=1, w0=2*np.pi/6)

# plot position and velocity of the sine position input
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
x = Xfunc_sine(t_eval, x0)
ax1.plot(t_eval, x)
v = derive_velocity(t_eval, Xfunc_sine, 0)
ax2.plot(t_eval[1:], v)
plt.show()

# solve the model using sine wave input
S = run_tof_model(scan_param, Xfunc_sine)

# plot model output
trvect_sim = scan_param['repetition_time'] * np.arange(scan_param['num_pulse'])
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(trvect_sim, S[:, 0:3])
ax.set_title('sine input')
plt.show()

