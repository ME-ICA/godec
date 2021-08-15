# godec

Go Decomposition in Python, with wrappers for working with fMRI data.

[![CircleCI](https://circleci.com/gh/ME-ICA/godec.svg?style=shield)](https://circleci.com/gh/ME-ICA/godec)
[![Codecov](https://codecov.io/gh/ME-ICA/godec/branch/main/graph/badge.svg?token=GEKDT6R0B7)](https://codecov.io/gh/ME-ICA/godec)

The core code in this package is adapted from three sources:
- [andrewssobral/godec](https://github.com/andrewssobral/godec):
  A Python translation of the standard algorithm.
- [prantikk/me-ica](https://bitbucket.org/prantikk/me-ica/src/v3/meica.libs/godec.py):
  A Python translation of the greedy semi-soft algorithm, for multi-echo fMRI data.
- [Go Decomposition](https://sites.google.com/site/godecomposition/code):
  The original MATLAB code for the standard and greedy semi-soft algorithms.
