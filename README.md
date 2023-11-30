# tof-model

in the "tof" folder:

"fresignal.py": two implementations of the main M equation. the _array version is what we're using right now because
it was slightly faster in most cases. 

"posfunclib.py": library of different position functions used to define the model input. for example if you want the input velocity to
be a sinousoid oscillation from v1 to v2 with initial position x0, you would use "compute_position_sine" which has the integral of such
a velocity (i.e. the x(t)) harcoded in the function. if you want to use timeseries velocity as input which we do with the PC data, then you
would use functions with "numeric" in the name. the "spatial" in the function names indicates if it will account for variations cross-sectional area
which is needed since the velocity becomes a function of both time and position in that case. for simulations using human PC data as input, we use
the compute_position_numeric_spatial.py function.

"tofmodel.py": main algorithm for the simulations. "run_tof_model.py" is the main routine that gets called. there is some code that implements
a lookup table feature (different from the lookup table that we used to use for contour maps in the inflow-analysis rep.). This feature saves solutions to a .pkl file so that
it can reuse them when possible. this actually doesn't improve performance that much so i usually dont use it. i may remove it later for clarity.
