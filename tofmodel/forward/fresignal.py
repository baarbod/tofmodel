# -*- coding: utf-8 -*-

import numpy as np
from numba import njit


# Define equation for flow-enhanced fMRI signal using array method (optimized)
@njit
def fre_signal_array(n, fa, tr, te, t1, t2, dt_list, offset_fact=1):
    m0 = 1
    sin_fa = np.sin(fa)
    cos_fa = np.cos(fa)
    exp_tr_t1 = np.exp(-tr / t1)
    exp_te_t2 = np.exp(-te / t2)
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
