"""The GODEC command-line interface."""

import argparse

from .. import godec
from ..utils import is_valid_file


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        dest="in_file",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="File to denoise with GODEC.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="Binary mask to apply to data.",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        help="Prefix for filenames generated.",
        default="",
    )
    parser.add_argument(
        "--method",
        dest="method",
        help="GODEC method.",
        default="greedy",
        choices=["greedy", "standard"],
    )
    parser.add_argument(
        "-r",
        "--ranks",
        dest="ranks",
        metavar="INT",
        type=int,
        nargs="+",
        help="Rank(s) of low rank component",
        default=[4],
    )
    parser.add_argument(
        "-k",
        "--increment",
        dest="drank",
        type=int,
        help="Rank search step size",
        default=2,
    )
    parser.add_argument(
        "-p",
        "--power",
        dest="inpower",
        type=int,
        help="Power for power method",
        default=2,
    )
    parser.add_argument(
        "-w",
        "--wavelet",
        dest="wavelet",
        help="Wavelet transform before GODEC",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        type=float,
        help="Threshold of some kind.",
        default=0.03,
    )
    parser.add_argument(
        "-n",
        "--norm_mode",
        dest="norm_mode",
        help=(
            "Normalization mode: variance normalization (vn), mean-centering (dm), "
            "or None (none)."
        ),
        default="vn",
        choices=["vn", "dm", "none"],
    )

    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    godec.run_godec_denoising(**vars(options))
