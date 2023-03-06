# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:15:57 2023

@author: Baarbod
"""

import numpy as np
from scipy.signal import square,sawtooth,triang
import matplotlib.pyplot as plt
from scipy.integrate import simps
import posfunclib as pfl
from functools import partial 

plt.close('all')

def define_resp_fourier(tmax, freq, N):
    samples=501
    # Generation of Triangular wave
    x= np.linspace(0,tmax,samples,endpoint=False)
    xp = tmax*freq*x
    y=triang(samples)
    
    # Fourier Coefficients
    a0=2./tmax*simps(y,x)
    an=lambda n:2.0/tmax*simps(y*np.cos(2.*np.pi*n*x/tmax),x)
    bn=lambda n:2.0/tmax*simps(y*np.sin(2.*np.pi*n*x/tmax),x)

    An = np.zeros(N)
    #An[0] = a0
    Bn = np.zeros(N)
    wn = np.zeros(N)
    for k in range(1, N+1, 1):
        An[k-1] = an(k)
        Bn[k-1] = bn(k)
        wn[k-1] = 2.*np.pi*k/tmax
        
    An = np.append([a0], An)    
    # Series sum
    s=a0/2.+sum([an(k)*np.cos(2.*np.pi*k*xp/tmax)+bn(k)*np.sin(2.*np.pi*
    k*xp/tmax) for k in range(1,N+1)])
    # Plotting

    plt.plot(x,s,label="Fourier series")
    plt.xlabel("$x$")
    plt.ylabel("$y=f(x)$")
    plt.legend(loc='best',prop={'size':10})
    plt.title("Triangular wave signal analysis by Fouries series")
    #plt.savefig("fs_triangular.png")
    plt.show()
    
    return An, Bn, wn

periods = 2
tperiod = 10 # Periodicity of the periodic function f(x)
freq = 0.2
N=10
An, Bn, w0 = define_resp_fourier(tperiod, freq, N)

t_eval = np.arange(0, periods*tperiod, 0.1)
Xfunc = partial(pfl.compute_position_resp, An=An, Bn=Bn, w0=w0)
x = Xfunc(t_eval, 0)
plt.figure()
plt.plot(t_eval, x)



    


