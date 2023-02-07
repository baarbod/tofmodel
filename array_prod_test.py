# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:38:51 2023

@author: bashe
"""

# test speed of np.prod when passing a matrix vs passing each column 
# in a for loop. Conclusion: array input is much faster.

import time
import numpy as np

def prod_array(array):
    p = np.prod(array, axis=0)
    return p

def prod_loop(array):
    num_rows, num_cols = array.shape
    p = np.zeros(num_cols)
    for icol in range(num_cols):
        p[icol] = np.prod(array[:, icol])
    return p    

a = np.tile(np.arange(1, 20), (10, 1))
b = a.transpose()

nrep = 100000

t = time.time()
for irep in range(nrep):
    p1 = prod_array(b)
    
elapsed = time.time() - t
print(elapsed)

t = time.time()
for irep in range(nrep):
    p2 = prod_loop(b)
elapsed = time.time() - t
print(elapsed)

    
    