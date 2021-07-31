"""Generate fixtures for tests."""
import os

import pytest
from nilearn import masking

from godec.tests.utils import get_test_data_path


@pytest.fixture(scope="session")
def testdata():
    """Load data from dataset into global variables."""
    test_path = get_test_data_path()
    func_file = os.path.join(
        test_path,
        "sub-04570_task-rest_echo-2_space-scanner_desc-partialPreproc_res-5mm_bold.nii.gz",
    )
    mask_file = os.path.join(
        test_path,
        "sub-04570_task-rest_space-scanner_desc-brain_res-5mm_mask.nii.gz",
    )
    data_array = masking.apply_mask(func_file, mask_file).T
    testdata = {
        "func": func_file,
        "mask": mask_file,
        "data_array": data_array,
    }
    return testdata
