# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:48:37 2023

@author: Baarbod
"""

import numpy as np
import operator
import functools

# array version can run faster for slower flows, but perform same or worse for 
# faster flows.

# Define equation for flow-enhanced fMRI signal
def fre_signal(n, fa, TR, T1, dt_list):
    
    M0 = 1
    C = np.cos(fa)
    Mzss = M0*(1 - np.exp(-TR/T1)) / (1 - np.exp(-TR/T1)*C)
    series = np.zeros(n-1)
    E = np.exp(-dt_list/T1)
    
    # try storing each series in matrix and do prod once at the end
    E_full = functools.reduce(operator.mul, E[1:n], 1)
    for m in range(n-1):
        num = 1 - E[n-m-1]
        # removed the -1 from indexing in order to include last element
        den = functools.reduce(operator.mul, E[1:n-m], 1)
        series[m] = C**(m) * num/den
                
    Mzn_pre = M0 * E_full * (np.sum(series) + C**(n-1))
        
    S = np.sin(fa)*(Mzn_pre - Mzss)
    
    return S

# Define equation for flow-enhanced fMRI signal usig array method
def fre_signal_array(n, fa, TR, T1, dt_list):
    
    M0 = 1
    C = np.cos(fa)
    Mzss = M0*(1 - np.exp(-TR/T1)) / (1 - np.exp(-TR/T1)*C)
    E = np.exp(-dt_list/T1)
    
    E_full = functools.reduce(operator.mul, E[1:n], 1)
    
    # put all terms in array
    full_array = np.ones((n-1, n-1))
    for m in range(n-1):
        full_array[m, 0:n-m-1] = E[1:n-m]       
        
    if n == 1:
        series = 0
        Mzn_pre = M0 * E_full * (np.sum(series) + C**(n-1))
        S = np.sin(fa)*(Mzn_pre - Mzss)
    else:    
        # call prod function on full array
        full_array_prod = functools.reduce(operator.mul, full_array)
        mm = np.arange(0, n-1)
        nummat = 1 - E[n - mm - 1]
        final_mat = C**mm * nummat/full_array_prod
        series = final_mat
        Mzn_pre = M0 * E_full * (np.sum(series) + C**(n-1))
        S = np.sin(fa)*(Mzn_pre - Mzss)
    
    return S

