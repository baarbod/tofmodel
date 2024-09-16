# -*- coding: utf-8 -*-

import numpy as np
import operator
import functools


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


# Define equation for flow-enhanced fMRI signal using array method
def fre_signal_array(n, fa, tr, t1, dt_list):
    m0 = 1
    c = np.cos(fa)
    mz_ss = m0 * (1 - np.exp(-tr / t1)) / (1 - np.exp(-tr / t1) * c)
    exponentials = np.exp(-dt_list / t1)

    exponentials_full = functools.reduce(operator.mul, exponentials[1:n], 1)

    # put all terms in array
    full_array = np.ones((n - 1, n - 1))
    for m in range(n - 1):
        full_array[m, 0:n - m - 1] = exponentials[1:n - m]

    if n == 1:
        series = 0
        mzn_pre = m0 * exponentials_full * (np.sum(series) + c ** (n - 1))
        s = np.sin(fa) * (mzn_pre - mz_ss)
    else:
        # call prod function on full array
        full_array_prod = functools.reduce(operator.mul, full_array.T)
        mm = np.arange(0, n - 1)
        nummat = 1 - exponentials[n - mm - 1]
        final_mat = c ** mm * nummat / full_array_prod
        series = final_mat
        mzn_pre = m0 * exponentials_full * (np.sum(series) + c ** (n - 1))
        s = np.sin(fa) * (mzn_pre - mz_ss)

    return s
