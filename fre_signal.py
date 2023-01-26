# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:48:37 2023

@author: Baarbod
"""

import numpy as np

# Define equation for flow-enhanced fMRI signal
def fre_signal(n, fa, TR, T1, dt_list):
    # e1 = math.exp(-TR/T1)
    # S = math.sin(fa)*(e1*math.cos(fa))**(n-1)*(1 - (1-e1)/(1-e1*math.cos(fa)))
    
    M0 = 1
    C = np.cos(fa)
    Mzss = M0*(1 - np.exp(-TR/T1)) / (1 - np.exp(-TR/T1)*C)
    positive_series = np.zeros(n)
    negative_series = np.zeros(n-1)
    E = np.exp(-dt_list[1:]/T1)
    
    if n == 1:
        Mzn_pre = M0*C
    else:
        for m in range(n):
            positive_series[m] = C**m * np.prod(E[0:n-1])/np.prod(E[0:n-m-1]) 
        for m in range(n-1):
            negative_series[m] = C**m * np.prod(E[0:n-1])/np.prod(E[0:n-m-2])
        Mzn_pre = M0 * (np.sum(positive_series) - np.sum(negative_series))
        
    S = np.sin(fa)*(Mzn_pre - Mzss)
    return S


