"""GODEC: A Python implementation of the Go Decomposition algorithm for fMRI data."""

from ._version import get_versions
from .decomposition import godec_fmri, godec_greedy_semisoft, godec_standard
from .due import Doi, due

__version__ = get_versions()["version"]

import warnings

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "godec_fmri",
    "godec_greedy_semisoft",
    "godec_standard",
    "__version__",
]

del get_versions
