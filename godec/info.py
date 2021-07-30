"""Base module variables."""
import importlib.util
import json
import os.path as op
from pathlib import Path

# Get version
spec = importlib.util.spec_from_file_location(
    "_version", op.join(op.dirname(__file__), "godec/_version.py")
)
_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version)

VERSION = _version.get_versions()["version"]
del _version

# Get list of authors from Zenodo file
with open(op.join(op.dirname(__file__), ".zenodo.json"), "r") as fo:
    zenodo_info = json.load(fo)
authors = [author["name"] for author in zenodo_info["creators"]]
author_names = []
for author in authors:
    if ", " in author:
        author_names.append(author.split(", ")[1] + " " + author.split(", ")[0])
    else:
        author_names.append(author)

# Get package description from README
# Since this file is executed from ../setup.py, the path to the README is determined by the
# location of setup.py.
readme_path = Path(__file__).parent.joinpath("README.md")
longdesc = readme_path.open().read()

# Fields
AUTHOR = "godec developers"
COPYRIGHT = "Copyright 2021, godec developers"
CREDITS = author_names
LICENSE = "GPL-2.0"
MAINTAINER = "Taylor Salo"
EMAIL = "tsalo006@fiu.edu"
STATUS = "Prototype"
URL = "https://github.com/me-ica/godec"
PACKAGENAME = "godec"
DESCRIPTION = (
    "A Python implementation of the Go Decomposition algorithm, adapted for fMRI data."
)
LONGDESC = longdesc

DOWNLOAD_URL = "https://github.com/ME-ICA/{name}/archive/{ver}.tar.gz".format(
    name=PACKAGENAME, ver=VERSION
)

REQUIRES = [
    "nilearn",
    "numpy>=1.15",
    "scipy>=1.3.3",
    "pywt>=1.1.1",
]

TESTS_REQUIRES = [
    "codecov",
    "coverage<5.0",
    "flake8>=3.7",
    "pytest",
    "pytest-cov",
    "requests",
]

EXTRA_REQUIRES = {
    "dev": ["versioneer"],
    "doc": [
        "sphinx>=1.5.3",
        "sphinx_rtd_theme",
        "sphinx-argparse",
    ],
    "tests": TESTS_REQUIRES,
    "duecredit": ["duecredit"],
}

ENTRY_POINTS = {"console_scripts": [
    "godec=godec.workflows.godec:_main",
]},

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES["all"] = list(set([v for deps in EXTRA_REQUIRES.values() for v in deps]))

# Supported Python versions using PEP 440 version specifiers
# Should match the same set of Python versions as classifiers
PYTHON_REQUIRES = ">=3.5"

# Package classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
