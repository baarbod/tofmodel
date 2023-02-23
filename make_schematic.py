# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:46:40 2023

@author: Baarbod
"""

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from functools import partial 
import posfunclib as pfl

plt.close('all')

def make_base_plot():
    fig, ax = plt.subplots()
    return fig, ax

def add_slice_shade(ax, w, islice, xr, c):    
    
    y = [(islice-1)*w, islice*w]
    x1 = 0
    x2 = xr
    ax.fill_betweenx(y, x1, x2, step='post', color=c, alpha=0.3)
    
# def draw_pulse_lines():
    

    
# def draw_trajectory():
    

fig, ax = make_base_plot()

w = 0.25
xr = 10
add_slice_shade(ax, w, 1, xr, 'C0')
add_slice_shade(ax, w, 2, xr, 'C1')
add_slice_shade(ax, w, 3, xr, 'C3')


Xfunc = partial(pfl.compute_position_sine, v1=-0.5, v2=0.5, w0=2*np.pi/5)
t_eval= np.arange(0, 10, 0.01)
x1 = Xfunc(t_eval, 0)
ax.plot(t_eval, x1)

x2 = Xfunc(t_eval, 0.25)
ax.plot(t_eval, x2)

x3 = Xfunc(t_eval, 0.5)
ax.plot(t_eval, x3)

trlist = np.arange(0, 10, 0.25)
for tr in trlist:
    line = lines.Line2D([tr, tr], [0, 0.75],
                        lw=1, alpha=0.5, color='black', axes=ax)
    ax.add_line(line)



