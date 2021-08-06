import os

from godec.workflows import godec as godec_workflow


def test_run_godec_denoising_standard_dm_cli_smoke(testdata, tmp_path_factory):
    """Smoke test godec.run_godec_denoising."""
    tmpdir = tmp_path_factory.mktemp("test_run_godec_denoising_standard_dm_cli_smoke")
    args = [
        "-d",
        testdata["func"],
        "-m",
        testdata["mask"],
        "--out-dir",
        str(tmpdir),
        "--prefix",
        "TEST",
        "--method",
        "standard",
        "--ranks",
        "2",
        "4",
        "6",
        "--norm_mode",
        "dm",
    ]
    godec_workflow._main(args)

    out_files = [
        "dataset_description.json",
        "TEST_desc-GODEC_rank-2_bold.nii.gz",
        "TEST_desc-GODEC_rank-2_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-2_errorts.nii.gz",
        "TEST_desc-GODEC_rank-4_bold.nii.gz",
        "TEST_desc-GODEC_rank-4_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-4_errorts.nii.gz",
        "TEST_desc-GODEC_rank-6_bold.nii.gz",
        "TEST_desc-GODEC_rank-6_lowrankts.nii.gz",
        "TEST_desc-GODEC_rank-6_errorts.nii.gz",
    ]
    for out_file in out_files:
        assert os.path.isfile(os.path.join(tmpdir, out_file))
