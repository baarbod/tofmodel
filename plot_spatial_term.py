# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:05:11 2023

@author: Baarbod
"""

import numpy as np
import matplotlib.pyplot as plt

def main(m=0):
   w = 0.25
   x = np.linspace(-2*w, 5*w)

   k = 1
   #m = 0.1
   r1 = 1

   pos_term = k*(r1/(m*x + r1))**2
   pos_term = np.heaviside(x, 0)*pos_term + np.heaviside(-x, 1) 

   # plot filtered signals
   fig, ax = plt.subplots(nrows=1, ncols=1)
   ax.plot(x, pos_term)
   ax.set_xlabel('Position (cm)', fontsize = 15)
   ax.set_ylabel('Scaling factor', fontsize = 15)
   plt.show()
   

if __name__ == "__main__":
   main()