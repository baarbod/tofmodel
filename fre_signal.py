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
    positive_series = np.zeros(n)
    negative_series = np.zeros(n-1)
    E = np.exp(-dt_list[1:]/T1)
    
    
    # CAN BE OPTIMIZED. WHY IS MATLAB FASTER? IS IT THE PRODUCT?
    num = functools.reduce(operator.mul, E[0:n-1], 1)
    
    if n == 1:
        Mzn_pre = M0*C
    else:
        for m in range(n):
            #num = functools.reduce(operator.mul, E[0:n-1], 1)
            den = functools.reduce(operator.mul, E[0:n-m-1], 1)
            positive_series[m] = C**m * num / den
            #positive_series[m] = C**m * np.prod(E[0:n-1])/np.prod(E[0:n-m-1]) 
            #positive_series[m] = C**m * math.prod(E[0:n-1])/np.prod(E[0:n-m-1]) 
            
        for m in range(n-1):
            #num = functools.reduce(operator.mul, E[0:n-1], 1)
            den = functools.reduce(operator.mul, E[0:n-m-2], 1)
            negative_series[m] = C**m * num / den            
            #negative_series[m] = C**m * np.prod(E[0:n-1])/np.prod(E[0:n-m-2])
            #negative_series[m] = C**m * math.prod(E[0:n-1])/np.prod(E[0:n-m-2])
            
        Mzn_pre = M0 * (np.sum(positive_series) - np.sum(negative_series))
        
    S = np.sin(fa)*(Mzn_pre - Mzss)
    return S

