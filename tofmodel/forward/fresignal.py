# -*- coding: utf-8 -*-

import numpy as np
import operator
import functools
from numba import njit


# array version can run faster for slower flows, but perform same or worse for
# faster flows.

# Define equation for flow-enhanced fMRI signal
def fre_signal(n, fa, tr, t1, dt_list):
    m0 = 1
    c = np.cos(fa)
    mz_ss = m0 * (1 - np.exp(-tr / t1)) / (1 - np.exp(-tr / t1) * c)
    series = np.zeros(n - 1)
    exponentials = np.exp(-dt_list / t1)

    # try storing each series in matrix and do prod once at the end
    exponentials_full = functools.reduce(operator.mul, exponentials[1:n], 1)
    for m in range(n - 1):
        num = 1 - exponentials[n - m - 1]
        # removed the -1 from indexing in order to include last element
        den = functools.reduce(operator.mul, exponentials[1:n - m], 1)
        series[m] = c ** m * num / den

    mzn_pre = m0 * exponentials_full * (np.sum(series) + c ** (n - 1))

    s = np.sin(fa) * (mzn_pre - mz_ss)

    return s


# Define equation for flow-enhanced fMRI signal using array method (optimized)
@njit
def fre_signal_array(n, fa, tr, t1, dt_list, offset_fact=1):
    
    # need to implement this properly at some point
    TE = 30/1000
    T2 = 1.5
    m0 = 1
    sin_fa = np.sin(fa)
    cos_fa = np.cos(fa)
    exp_tr_t1 = np.exp(-tr / t1)
    exp_te_t2 = np.exp(-TE / T2)
    mz_ss = offset_fact * m0 * (1 - exp_tr_t1) / (1 - exp_tr_t1 * cos_fa)
    
    exponentials = np.empty(n)
    for i in range(n):
        exponentials[i] = np.exp(-dt_list[i] / t1)

    exponentials_full = 1.0
    for i in range(1, n):
        exponentials_full *= exponentials[i]

    if n == 1:
        mzn_pre = m0 * exponentials_full
        s = sin_fa * exp_te_t2 * (mzn_pre - mz_ss)
    else:
        series_sum = 0.0
        for mm in range(n - 1):
            num = 1.0 - exponentials[n - mm - 1] 
            prod = 1.0
            for k in range(1, n - mm):
                prod *= exponentials[k]
            series_sum += (cos_fa ** mm) * num / prod

        mzn_pre = m0 * exponentials_full * (series_sum + cos_fa ** (n - 1))
        s = sin_fa * exp_te_t2 * (mzn_pre - mz_ss)

    return s
