# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm


# PARAMETERS
tr = 0.5 # repetition time [s]
te = 0.025 # echo time [s]
npulse = 200 # number of TR cycles
w = 0.5 # slice thickness [cm]
fa = 45 # flip angle [degrees]
t1 = 4.0 # T1 relaxation time constant for CSF [s]
t2 = 1.5 # T2 relaxation time constant for CSF [s]
nslice = 10 # number of imaging slices
MBF = 2 # multiband factor
alpha_list = [0, 0.2, 0.07, 0.3, 0.15, 
              0, 0.2, 0.07, 0.3, 0.15] # slice excitation timings [s]

# INPUT AREA DEFINITION (STRAIGHT TUBE)
xarea = np.linspace(-3, 3, 1000) # positions [cm]
area = np.ones(xarea.size) # cross-sectional areas [cm^2]

# INPUT VELOCITY DEFINITION (SINUSOIDAL FUNCTION)
t = np.arange(0, npulse*tr, tr/100) # time vector for velocity time-series [s]
f = 0.1 # flow frequency [Hz]
a = 1.0 # velocity amplitude [cm/s]
v = a*np.sin(2*np.pi*f*t) # velocity time-series [cm/s]

# RUN SIMULATION
x_func_area = partial(pfl.compute_position_numeric_spatial, tr_vect=t, vts=v, xarea=xarea, area=area)
signal = tm.simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, alpha_list, MBF, x_func_area, multithread=False)
tr_vect = tr*np.arange(0, signal.shape[0])

# PLOTTING FIRST 3 SLICES
fig, ax = plt.subplots()
ax.plot(tr_vect, signal[:, :4])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Inflow Signal (a.u.)')