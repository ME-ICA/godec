"""Core GODEC functions."""
import json
import logging
import os

import numpy as np
from nilearn.masking import apply_mask, unmask
from scipy.linalg import qr
from scipy.sparse.linalg import svds
from sklearn import metrics

from . import references
from .due import due
from .utils import dwtmat, idwtmat, wthresh

LGR = logging.getLogger(__name__)


@due.dcite(
    references.GODEC,
    description="Introduces the standard GODEC algorithm.",
)
def standard_godec(
    X,
    rank=2,
    card=None,
    iterated_power=1,
    max_iter=100,
    tol=0.001,
    quiet=False,
):
    """Run the standard GODEC method.

    Default threshold of .03 is assumed to be for input in the range 0-1...
    original matlab had 8 out of 255, which is about .03 scaled to 0-1 range

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        Data matrix. If ``n_samples`` < ``n_features``, the matrix will automatically be
        transposed prior to decomposition.
        Outputs will then be transposed to ensure compatible dimensions.
    rank : :obj:`int`, optional
        The rank of low-rank matrix. Must be an integer >= 1.
        The default is 2.
    card : :obj:`int` or None, optional
        The cardinality of the sparse matrix. Must be an integer >= 0, or None,
        in which case it will be set to the number of elements in X.
        Default is None.
    iterated_power : :obj:`int`, optional
        Number of iterations for the power method, increasing it leads to better accuracy and more
        time cost. Must be an integer >= 1. The default is 1.
    max_iter : :obj:`int`, optional
        Maximum number of iterations to be run. Must be an integer >= 1. The default is 100.
    tol : :obj:`float`, optional
        Tolerance for stopping criteria. Must be a float > 0. The default is 0.001.

    Returns
    -------
    low_rank : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The low-rank matrix. Known as L in the original code and formulae.
    sparse : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The sparse matrix. Known as S in the original code and formulae.
    reconstruction : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The reconstruction matrix. Known as LS in the original code and formulae.
    noise : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The noise matrix. Known as G in the original code and formulae.
    rmse : :obj:`numpy.ndarray` of shape (n_iterations,)
        Root mean-squared error values. One value for each iteration.

    Notes
    -----
    From https://github.com/andrewssobral/godec.

    Minimal variable name changes have been made by Taylor Salo for readability and PEP8
    compatibility.

    License
    -------
    MIT License

    Copyright (c) 2020 Andrews Sobral

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    rmse = []
    card = np.prod(X.shape) if card is None else card

    data_transposed = False
    if X.shape[0] < X.shape[1]:
        LGR.info("Data were transposed prior to decomposition.")
        data_transposed = True
        X = X.T

    _, n_features = X.shape

    LGR.info("++Starting Go Decomposition")

    # Initialization of L and S
    low_rank = X.copy()
    sparse = np.zeros(X.shape)
    reconstruction = np.zeros(X.shape)

    for i_iter in range(max_iter):
        # Update of L
        Y2 = np.random.randn(n_features, rank)
        for j_power in range(iterated_power):
            Y1 = low_rank.dot(Y2)
            Y2 = low_rank.T.dot(Y1)

        Q, R = qr(Y2, mode="economic")
        low_rank_new = (low_rank.dot(Q)).dot(Q.T)

        # Update of S
        T = low_rank - low_rank_new + sparse
        low_rank = low_rank_new
        T_1d = T.reshape(-1)
        sparse_1d = sparse.reshape(-1)
        idx = np.abs(T_1d).argsort()[::-1]
        sparse_1d[idx[:card]] = T_1d[idx[:card]]
        sparse = sparse_1d.reshape(sparse.shape)

        # Reconstruction
        reconstruction = low_rank + sparse

        # Stopping criteria
        error = np.sqrt(metrics.mean_squared_error(X, reconstruction))
        rmse.append(error)

        if not quiet:
            print(f"Iteration: {i_iter}, RMSE: {error}")

        if error <= tol:
            break

    LGR.debug(f"Finished at iteration {i_iter}")

    noise = X - reconstruction
    if data_transposed:
        low_rank = low_rank.T
        sparse = sparse.T
        reconstruction = reconstruction.T
        noise = noise.T

    rmse = np.array(rmse)

    return low_rank, sparse, reconstruction, noise, rmse


@due.dcite(
    references.GODEC,
    description="Introduces the semi-soft GODEC algorithm.",
)
@due.dcite(
    references.BILATERAL_SKETCH,
    description="Introduces the greedy bilateral smoothing method.",
)
def greedy_semisoft_godec(D, rank, tau=1, tol=1e-7, iterated_power=2, rank_step_size=2, quiet=False):
    """Run the Greedy Semi-Soft GoDec Algorithm (GreBsmo).

    Parameters
    ----------
    D : array
        nxp data matrix with n samples and p features
    ranks : list of int
        rank(L)<=rank
    tau : float
        soft thresholding
    iterated_power : float
        >=0, power scheme modification, increasing it lead to better accuracy and more time cost
    rank_step_size : int
        rank stepsize. Was k.

    Returns
    -------
    low_rank : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The low-rank matrix. Known as L in the original code and formulae.
    sparse : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The sparse matrix. Known as S in the original code and formulae.
    reconstruction : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The reconstruction matrix. Known as LS in the original code and formulae.
    noise : :obj:`numpy.ndarray` of shape (n_samples, n_features)
        The noise matrix. Known as G in the original code and formulae.
    error : :obj:`numpy.ndarray` of shape (n_iterations,)
        ||X-L-S||/||X||

    Notes
    -----
    Translated to Python from GreGoDec.m and adapted somewhat.

    References
    ----------
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix Decomposition in
        Noisy Case", ICML 2011
    Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and Smoothing", AISTATS 2013.

    Tianyi Zhou, 2013, All rights reserved.
    """
    data_transposed = False
    if D.shape[0] < D.shape[1]:
        LGR.info("Data were transposed prior to decomposition.")
        data_transposed = True
        D = D.T

    n_samples, n_features = D.shape

    # To match MATLAB's norm on a matrix, you need an order of 2.
    norm_D = np.linalg.norm(D, ord=2)

    # initialization of L and S by discovering a low-rank sparse SVD and recombining
    assert rank % rank_step_size == 0, "Rank must be divisible by step size."
    n_rank_steps = rank // rank_step_size
    rank_steps = np.arange(rank_step_size, rank + rank_step_size, rank_step_size)

    error = np.zeros((n_rank_steps * iterated_power) + 1)

    X, s, Y = svds(D, rank_step_size, which="LM")
    s = np.diag(s)

    X = X.dot(s)
    low_rank = X.dot(Y)
    sparse = wthresh(D - low_rank, tau)
    T = D - low_rank - sparse

    error[0] = np.linalg.norm(T, ord=2) / norm_D
    iteration_counter = 0  # was named iii (very useful)
    error_counter = 0
    stop = False

    # Define some variables that shouldn't be touched before they're updated.
    X1 = Y1 = L1 = S1 = T1 = None

    for i_rank_step, rank_step in enumerate(rank_steps):
        # parameters for alf
        rrank = rank
        est_rank = 1
        rank_min = 1
        rk_jump = 10
        alf = 0
        increment = 1
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50
        if iteration_counter == iterated_power * (i_rank_step - 2) + 1:
            print(
                f"Changing iteration counter from {iteration_counter} to "
                f"{iteration_counter + iterated_power}"
            )
            iteration_counter = iteration_counter + iterated_power

        for j_iter in range(iterated_power):
            if not quiet:
                print(f"rank_step: {rank_step}, iteration: {j_iter}, rrank: {rrank}, alf: {alf}")

            # Update of X
            X = low_rank.dot(Y.T)

            # The original MATLAB code has 3 outputs for qr(X, 0) (including E) when est_rank==1.
            # However, E is never used, so this doesn't seem to matter.
            X, R = qr(X, mode="economic")
            # CHECK qr output formats

            # Update of Y
            Y = X.T.dot(low_rank)
            low_rank = X.dot(Y)

            # Update of S
            T = D - low_rank
            sparse = wthresh(T, tau)

            # Error, stopping criteria
            T = T - sparse
            # error_counter = iteration_counter + j_iter + 1  # was ii
            error_counter += 1
            # ok

            error[error_counter] = np.linalg.norm(T, ord=2) / norm_D
            if error[error_counter] < tol:
                stop = True
                print("Error below tolerance. Stopping early.")
                break

            # adjust est_rank
            if est_rank == 1:
                # This was the subfunction rank_estimator_adaptive
                # if/else added because MATLAB will work with empty arrays without raising errors
                if R.size > 1:
                    dR = np.abs(np.diag(R))
                    drops = dR[:-1] / dR[1:]
                    dmx = max(drops)
                    imx = np.argmax(drops)
                    rel_drp = (rank - 1) * dmx / (sum(drops) - dmx)
                else:
                    rel_drp = 0

                if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or (
                    itr_rank > maxitr_reduce_rank
                ):
                    rrank = np.maximum(imx, np.floor(0.1 * rank), rank_min)
                    # res and normz are not defined/set in the MATLAB code
                    # error[error_counter] = np.linalg.norm(res) / normz
                    est_rank = 0
                    itr_rank = 0

            if rrank != rank:
                rank = rrank
                if est_rank == 0:
                    alf = 0
                    print("rrank != rank. Whatever that means.")
                    continue

            # adjust alf
            ratio = error[error_counter] / error[error_counter - 1]
            if np.isinf(ratio):
                LGR.warning("Infinite error ratio.")
                ratio = 0

            if ratio >= 1.1:
                increment = np.maximum(0.1 * alf, 0.1 * increment)
                X = X1
                Y = Y1
                low_rank = L1
                sparse = S1
                T = T1
                error[error_counter] = error[error_counter - 1]
                alf = 0
            elif ratio > 0.7:
                increment = np.maximum(increment, 0.25 * alf)
                alf = alf + increment

            # Update of L
            X1 = X.copy()
            Y1 = Y.copy()
            L1 = low_rank.copy()
            S1 = sparse.copy()
            T1 = T.copy()

            low_rank = low_rank + ((1 + alf) * (T))

            # Add coreset
            if (j_iter > 7) and (
                np.mean(error[error_counter - 7 : error_counter]) / error[error_counter - 8] > 0.92
            ):
                iteration_counter = error_counter
                sf = X.shape[1]
                if Y.shape[0] - sf >= rank_step_size:
                    Y = Y[:sf, :]

                print(
                    "Detecting average decrease in error of <= 8% over past 7 iterations, "
                    "indicating stabilization (I guess). Stopping early."
                )
                break

        if stop:
            print("Stopping early.")
            break

        # Coreset
        if i_rank_step < n_rank_steps - 1:
            v = np.random.randn(rank_step_size, n_samples).dot(low_rank)
            Y = np.vstack([Y, v])

    # Remove unused elements from error vector
    error = np.trim_zeros(error, trim="b")

    low_rank = X.dot(Y)
    reconstruction = low_rank + sparse
    noise = D - reconstruction
    if data_transposed:
        low_rank = low_rank.T
        sparse = sparse.T
        reconstruction = reconstruction.T
        noise = noise.T

    return low_rank, sparse, reconstruction, noise, error


def run_godec_denoising(
    in_file,
    mask,
    out_dir=".",
    prefix="",
    method="greedy",
    ranks=[4],
    norm_mode="vn",
    drank=2,
    iterated_power=2,
    wavelet=False,
):
    """Run GODEC denoising in neuroimaging data.

    Notes
    -----
    - Prantik mentioned that GODEC is run on outputs (e.g., High-Kappa), not inputs.
      https://github.com/ME-ICA/me-ica/issues/4#issuecomment-369058732
    - The paper tested ranks of 1-4. See page 5 of online supplemental methods.
    - The paper used a discrete Daubechies wavelet transform before and after GODEC,
      with rank-1 approximation and 100 iterations. See page 4 of online supplemental methods.
    """
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    masked_data = apply_mask(in_file, mask)

    # Transpose to match ME-ICA convention (SxT instead of TxS)
    masked_data = masked_data.T

    if norm_mode == "dm":
        # Demean
        rmu = masked_data.mean(-1)
        dnorm = masked_data - rmu[:, np.newaxis]
    elif norm_mode == "vn":
        rmu = masked_data.mean(-1)
        rstd = masked_data.std(-1)
        dnorm = (masked_data - rmu[:, np.newaxis]) / rstd[:, np.newaxis]
    else:
        dnorm = masked_data

    # GoDec
    godec_outputs = {}
    if wavelet:
        LGR.info("++Wavelet transforming data")
        temp_data, cal = dwtmat(dnorm)
    else:
        temp_data = dnorm.copy()

    for rank in ranks:
        if method == "greedy":
            # GreGoDec
            lowrank, sparse, reconstruction, noise, error = greedy_semisoft_godec(
                temp_data,
                rank=rank,
                tau=1,
                tol=1e-7,
                iterated_power=iterated_power,
                rank_step_size=drank,
            )
        else:
            for rank in ranks:
                lowrank, sparse, reconstruction, noise, error = standard_godec(
                    temp_data,
                    rank=rank,
                    card=None,
                    iterated_power=iterated_power,
                    tol=0.001,
                    max_iter=500,
                )

        godec_outputs[rank] = {
            "lowrank": lowrank,
            "sparse": sparse,
            "reconstruction": reconstruction,
            "noise": noise,
        }

    if wavelet:
        LGR.info("++Inverse wavelet transforming outputs")
        for rank in godec_outputs.keys():
            godec_outputs[rank] = [idwtmat(arr, cal) for arr in godec_outputs[rank]]

    if norm_mode == "dm":
        for rank in godec_outputs.keys():
            # Just add mean back into low-rank matrix
            godec_outputs[rank]["lowrank"] = godec_outputs[rank]["lowrank"] + rmu[:, np.newaxis]
            godec_outputs[rank]["reconstruction"] = (
                godec_outputs[rank]["reconstruction"] + rmu[:, np.newaxis]
            )
    elif norm_mode == "vn":
        for rank in godec_outputs.keys():
            # Low-rank and reconstructed matrices get variance and mean added back in
            godec_outputs[rank]["lowrank"] = (
                godec_outputs[rank]["lowrank"] * rstd[:, np.newaxis]
            ) + rmu[:, np.newaxis]
            godec_outputs[rank]["reconstruction"] = (
                godec_outputs[rank]["reconstruction"] * rstd[:, np.newaxis]
            ) + rmu[:, np.newaxis]
            # Other matrices are just rescaled by variance
            godec_outputs[rank]["sparse"] = godec_outputs[rank]["sparse"] * rstd[:, np.newaxis]
            godec_outputs[rank]["noise"] = godec_outputs[rank]["noise"] * rstd[:, np.newaxis]

    metadata = {
        "normalization": norm_mode,
        "wavelet": wavelet,
        "ranks": ranks,
        "rank_step_size": drank,
        "iterated_power": iterated_power,
    }
    metadata_file = os.path.join(out_dir, "dataset_description.json")
    with open(metadata_file, "w") as fo:
        json.dump(metadata, fo, sort_keys=True, indent=4)

    for rank, outputs in godec_outputs.items():
        lowrank_img = unmask(outputs[0].T, mask)
        sparse_img = unmask(outputs[1].T, mask)
        reconstruction_img = unmask(outputs[2].T, mask)
        noise_img = unmask(outputs[3].T, mask)

        lowrank_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_lowrankts.nii.gz")
        )
        sparse_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_bold.nii.gz")
        )
        reconstruction_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODECReconstructed_rank-{rank}_bold.nii.gz")
        )
        noise_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_errorts.nii.gz")
        )
