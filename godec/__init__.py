"""
mapca: A Python implementation of the moving average principal components analysis methods from
GIFT.
"""

from ._version import get_versions
from .due import Doi, due
from .godec import greedy_semisoft_godec, run_godec_denoising, standard_godec

__version__ = get_versions()["version"]

import warnings

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "run_godec_denoising",
    "standard_godec",
    "greedy_semisoft_godec",
    "__version__",
]

del get_versions
