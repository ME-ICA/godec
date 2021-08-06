"""Tests for the godec.godec module."""
import os

from godec import godec


def test_run_godec_denoising_smoke(testdata, tmp_path_factory):
    """Smoke test godec.run_godec_denoising."""
    tmpdir = tmp_path_factory.mktemp("test_run_godec_denoising_smoke")
    godec.run_godec_denoising(
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


def test_run_godec_denoising_smoke_wavelet(testdata, tmp_path_factory):
    """Smoke test godec.run_godec_denoising with wavelet transformation."""
    tmpdir = tmp_path_factory.mktemp("test_run_godec_denoising_smoke_wavelet")
    godec.run_godec_denoising(
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


def test_greedy_semisoft_godec_smoke(testdata):
    """Smoke test godec.greedy_semisoft_godec."""
    out = godec.greedy_semisoft_godec(
        testdata["data_array"],
        rank=2,
        tau=1,
        tol=1e-7,
        iterated_power=2,
        rank_step_size=1,
    )
    assert len(out) == 5
    assert all(val.shape == testdata["data_array"].shape for val in out[:-1])


def test_standard_godec_smoke(testdata):
    """Smoke test godec.standard_godec."""
    out = godec.standard_godec(
        testdata["data_array"],
        rank=2,
        card=None,
        iterated_power=1,
        max_iter=100,
        tol=1e-3,
    )
    assert len(out) == 5
    assert all(val.shape == testdata["data_array"].shape for val in out[:-1])
