
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

# --- PARAMETERS ---
tr = 0.5            # Repetition time [s]
te = 0.025          # Echo time [s]
npulse = 100        # Number of TR cycles
w = 0.25             # Slice thickness [cm]
fa = 45             # Flip angle [degrees]
t1 = 4.0            # T1 relaxation time constant for CSF [s]
t2 = 1.5            # T2 relaxation time constant for CSF [s]
nslice = 10         # Number of imaging slices
mbf = 2             # Multiband factor

# Slice excitation timings [s]
alpha_list = [0, 0.2, 0.07, 0.3, 0.15, 0, 0.2, 0.07, 0.3, 0.15]

# --- INPUT AREA DEFINITION (STRAIGHT TUBE) ---
xarea = np.linspace(-3, 3, 1000)  # Positions [cm]
area = np.ones(xarea.size)        # Cross-sectional areas [cm^2]

# --- INPUT VELOCITY DEFINITION (SINUSOIDAL FUNCTION) ---
t = np.arange(0, npulse * tr, tr / 100)  # Time vector for velocity time-series [s]

# Flow frequencies [Hz] and amplitudes [cm/s]
f1, a1 = 0.05, 0.1
f2, a2 = 0.15, 0.2
f3, a3 = 1.00, 0.2

v = (a1 * np.sin(2 * np.pi * f1 * t) + 
        a2 * np.sin(2 * np.pi * f2 * t) + 
        a3 * np.sin(2 * np.pi * f3 * t) + 0.05)

# Plot velocity profile
fig1, ax1 = plt.subplots()
ax1.plot(t, v, color='black')
ax1.axhline(0, color='red', linestyle='--', linewidth=2.0)
ax1.set_title('Input Velocity Time-Series')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Velocity (cm/s)')

# --- RUN SIMULATION ---
print("Starting simulation...")
tstart = time.time()

x_func_area = partial(pfl.compute_position_numeric_spatial, 
                        tr_vect=t, vts=v, xarea=xarea, area=area)
                        
signal = tm.simulate_inflow(tr, te, npulse, w, fa, t1, t2, nslice, 
                            alpha_list, mbf, x_func_area, ncpu=1)
                            
print(f"Total simulation time: {time.time() - tstart:.2f} seconds")

# --- PLOTTING RESULTS ---
tr_vect = tr * np.arange(0, signal.shape[0])

# Plotting signal in the first 4 slices
fig2, ax2 = plt.subplots()
ax2.plot(tr_vect, signal[:, :4])
ax2.set_title('Inflow Signal in First 4 Slices')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Inflow Signal (a.u.)')
ax2.legend([f'Slice {i+1}' for i in range(4)])

# Display the generated plots
plt.show()
