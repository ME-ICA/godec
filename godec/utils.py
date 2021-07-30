"""Miscellaneous utility functions for godec."""
import os

import numpy as np
import pywt


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if (arg is not None) and (not os.path.isfile(arg)):
        parser.error(f"The file {arg} does not exist!")

    return arg


def dwtmat(mmix):
    """Apply a discrete wavelet transform to a matrix."""
    lt = len(np.hstack(pywt.dwt(mmix[0], "db2")))
    mmix_wt = np.zeros([mmix.shape[0], lt])
    for ii in range(mmix_wt.shape[0]):
        wtx = pywt.dwt(mmix[ii], "db2")
        cAlen = len(wtx[0])
        mmix_wt[ii] = np.hstack(wtx)
    return mmix_wt, cAlen


def idwtmat(mmix_wt, cAl):
    """Apply a discrete inverse wavelet transform to a matrix."""
    lt = len(pywt.idwt(mmix_wt[0, :cAl], mmix_wt[0, cAl:], "db2"))
    mmix_iwt = np.zeros([mmix_wt.shape[0], lt])
    for ii in range(mmix_iwt.shape[0]):
        mmix_iwt[ii] = pywt.idwt(mmix_wt[ii, :cAl], mmix_wt[ii, cAl:], "db2")
    return mmix_iwt


def wthresh(a, thresh):
    """Determine soft wavelet threshold."""
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)
