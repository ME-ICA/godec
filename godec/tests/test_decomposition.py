"""Tests for the godec.decomposition module."""
import os

import nibabel as nib

from godec import decomposition


def test_godec_fmri_smoke(testdata, tmp_path_factory):
    """Smoke test decomposition.godec_fmri."""
    tmpdir = tmp_path_factory.mktemp("test_godec_fmri_smoke")
    decomposition.godec_fmri(
        testdata["func"],
        testdata["mask"],
        out_dir=tmpdir,
        prefix="TEST",
        method="greedy",
        ranks=[2, 6],
        norm_mode="vn",
        iterated_power=2,
        rank_step_size=2,
        wavelet=False,
    )
    out_files = [
        "dataset_description.json",
        "TEST_desc-GODEC_rank-2_bold.nii.gz",
        "TEST_desc-GODEC_rank-2_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-2_errorts.nii.gz",
        "TEST_desc-GODECReconstructed_rank-2_bold.nii.gz",
        "TEST_desc-GODEC_rank-6_bold.nii.gz",
        "TEST_desc-GODEC_rank-6_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-6_errorts.nii.gz",
        "TEST_desc-GODECReconstructed_rank-6_bold.nii.gz",
    ]
    for out_file in out_files:
        assert os.path.isfile(os.path.join(tmpdir, out_file))

    orig_img = nib.load(testdata["func"])
    for out_file in out_files:
        if out_file.endswith(".nii.gz"):
            test_img = nib.load(os.path.join(tmpdir, out_file))
            assert test_img.shape == orig_img.shape, (
                f"{out_file}: {orig_img.shape} != {test_img.shape}"
            )


def test_godec_fmri_smoke_wavelet(testdata, tmp_path_factory):
    """Smoke test decomposition.godec_fmri with wavelet transformation."""
    tmpdir = tmp_path_factory.mktemp("test_godec_fmri_smoke_wavelet")
    decomposition.godec_fmri(
        testdata["func"],
        testdata["mask"],
        out_dir=tmpdir,
        prefix="TEST",
        method="greedy",
        ranks=[2],
        norm_mode="vn",
        iterated_power=2,
        rank_step_size=2,
        wavelet=True,
    )
    out_files = [
        "dataset_description.json",
        "TEST_desc-GODEC_rank-2_bold.nii.gz",
        "TEST_desc-GODEC_rank-2_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-2_errorts.nii.gz",
        "TEST_desc-GODECReconstructed_rank-2_bold.nii.gz",
    ]
    for out_file in out_files:
        assert os.path.isfile(os.path.join(tmpdir, out_file))

    orig_img = nib.load(testdata["func"])
    for out_file in out_files:
        if out_file.endswith(".nii.gz"):
            test_img = nib.load(os.path.join(tmpdir, out_file))
            assert test_img.shape == orig_img.shape, (
                f"{out_file}: {orig_img.shape} != {test_img.shape}"
            )


def test_godec_greedy_semisoft_smoke(testdata):
    """Smoke test decomposition.godec_greedy_semisoft."""
    out = decomposition.godec_greedy_semisoft(
        testdata["data_array"],
        rank=2,
        tau=1,
        tol=1e-7,
        iterated_power=2,
        rank_step_size=1,
    )
    assert len(out) == 5
    assert all(val.shape == testdata["data_array"].shape for val in out[:-1])


def test_godec_standard_smoke(testdata):
    """Smoke test decomposition.godec_standard."""
    out = decomposition.godec_standard(
        testdata["data_array"],
        rank=2,
        card=None,
        iterated_power=1,
        max_iter=100,
        tol=1e-3,
    )
    assert len(out) == 5
    assert all(val.shape == testdata["data_array"].shape for val in out[:-1])
