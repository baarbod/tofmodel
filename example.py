# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tof import posfunclib as pfl
from tof import tofmodel as tm
from functools import partial

plt.close('all')

# define model scan parameters
scan_param = {
    'slice_width': 0.25,        # cm
    'repetition_time': 0.504,   # seconds
    'flip_angle': 45,           # degrees
    't1_time': 4,               # seconds
    'num_slice': 21,
    'num_pulse': 200,
    'MBF': 3,                   # number of slice groups
    'alpha_list': [
                0, 0.2775, 0.07, 0.345, 0.1375, 0.415, 0.2075,
                0, 0.2775, 0.07, 0.345, 0.1375, 0.415, 0.2075,
                0, 0.2775, 0.07, 0.345, 0.1375, 0.415, 0.2075
                    ] # excitation timings for every slice, from .json 
                      # file obtained from unpacking dicoms
    }

tr = scan_param['repetition_time']

# define input velocity as a 0.1 Hz sinusoid
v1 = -0.4 # cm/s
v2 = 0.4 # cm/s
frequency = 0.1 # Hz 
# use partial to make the position function depend only on time and initial position
x_func = partial(pfl.compute_position_sine, v1=v1, v2=v2, w0=2*np.pi*frequency)

# plot position vs time of an example proton starting at x=0
t = tr * np.arange(1, scan_param['num_pulse'] + 1)
fig, ax = plt.subplots()
x0 = 0
ax.plot(t, x_func(t, x0))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (cm)')

# solve the model
s = tm.run_tof_model(scan_param, x_func, progress=True)
s_first_three_slices = s[:, 0:3]

# plot model output signal
fig, ax = plt.subplots()
ax.plot(t, s_first_three_slices)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Simulated Inflow Signal')


