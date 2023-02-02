# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:48:37 2023

@author: Baarbod
"""

import numpy as np
#import math 
import operator
import functools


# Define equation for flow-enhanced fMRI signal
def fre_signal(n, fa, TR, T1, dt_list):
    # e1 = math.exp(-TR/T1)
    # S = math.sin(fa)*(e1*math.cos(fa))**(n-1)*(1 - (1-e1)/(1-e1*math.cos(fa)))
    
    
    M0 = 1
    C = np.cos(fa)
    Mzss = M0*(1 - np.exp(-TR/T1)) / (1 - np.exp(-TR/T1)*C)
    series = np.zeros(n-1)
    E = np.exp(-dt_list/T1)
    
    
    E_full = functools.reduce(operator.mul, E[1:n], 1)
    for m in range(n-1):
        num = 1 - E[n-m-1]
        # removed the -1 from indexing in order to include last element
        den = functools.reduce(operator.mul, E[1:n-m], 1)
        series[m] = C**(m) * num/den
    
    Mzn_pre = M0 * E_full * (np.sum(series) + C**(n-1))
        
    S = np.sin(fa)*(Mzn_pre - Mzss)
    
    return S

