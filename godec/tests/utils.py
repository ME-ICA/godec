"""Utility functions for testing godec."""
from os.path import abspath, dirname, join, sep

import numpy as np
from nilearn import masking


def get_test_data_path():
    """Return the path to test datasets.

    Test-related data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return abspath(join(dirname(__file__), "data") + sep)


def create_mask():
    """Create a reduced mask for testing."""
    data_dir = get_test_data_path()

    in_file = join(
        data_dir,
        "sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_res-5mm_bold.nii.gz",
    )
    mask_file = join(data_dir, "sub-04570_task-rest_space-scanner_desc-brain_res-5mm_mask.nii.gz")

    mask = masking.compute_epi_mask(in_file, exclude_zeros=True)

    # Zero out any voxels where the SD is zero over time
    data = masking.apply_mask(in_file, mask)
    mask_data = masking.apply_mask(mask, mask)
    mask_data[np.std(data, axis=0) == 0] = 0

    mask2 = masking.unmask(mask_data, mask)
    mask2.to_filename(mask_file)
