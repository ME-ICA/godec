"""Core GODEC functions."""
import json
import logging
import os

import numpy as np
from nilearn.masking import apply_mask, unmask
from scipy.linalg import qr
from scipy.sparse.linalg import svds

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
    tol=0.001,
    max_iter=100,
    random_seed=0,
    thresh=0.03,
):
    """Run the standard GODEC method.

    Default threshold of .03 is assumed to be for input in the range 0-1...
    original matlab had 8 out of 255, which is about .03 scaled to 0-1 range

    Parameters
    ----------
    X : numpy.ndarray
        nxp data matrix with n samples and p features
    rank : int
        rank(L)<=rank
    card : :obj:`int` or None, optional
        The cardinality of the sparse matrix. Must be an integer >= 0, or None,
        in which case it will be set to the number of elements in X.
        Default is None.
    iterated_power : int, optional
        Number of iterations for the power method, increasing it lead to better accuracy and more
        time cost. The default is 1.
    tol : float
        Error bound
    thresh : float
        soft thresholding

    Notes
    -----
    From SSGoDec.m
    """
    LGR.info("++Starting Go Decomposition")
    m, n = X.shape
    if m < n:
        X = X.T

    L = X.copy()
    S = np.zeros(L.shape)
    itr = 0
    rmse = []
    random_state = np.random.RandomState(random_seed)
    while True:
        # Update of L
        Y2 = random_state.randn(np.minimum(m, n), rank)
        for i in range(iterated_power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1)

        Q, R = qr(Y2, mode="full")
        L_new = np.dot(np.dot(L, Q), Q.T)

        # Update of S
        T = L - L_new + S
        L = L_new

        # The version from the original GODEC code
        # Requires "card" (cardinality of S)
        # card = 310000
        # T = np.ravel(T)
        # S = np.zeros(T.shape)
        # idx = np.argsort(np.abs(T))[::-1]
        # S[idx[:card]] = T[idx[:card]]
        # S = np.reshape(S, X.shape)
        # T[idx[:card]] = 0
        # T = np.reshape(T, X.shape)

        # Alternate approach taken from the Greedy Semi-Soft GoDec Algorithm code
        # This automatically determines cardinality of S from tau threshold.
        S = wthresh(T, thresh)
        T = T - S

        # Error, stopping criteria
        rmse.append(np.linalg.norm(T))
        if (rmse[-1] < tol) or (itr >= max_iter):
            break

        L += T
        itr += 1

    error = np.linalg.norm((L + S) - X) / np.linalg.norm(X)
    if m < n:
        L = L.T
        S = S.T

    LGR.debug(f"Finished at iteration {itr}")

    return L, S, error, rmse


@due.dcite(
    references.GODEC,
    description="Introduces the semi-soft GODEC algorithm.",
)
def greedy_semisoft_godec(D, ranks, tau=1, tol=1e-7, inpower=2, k=2):
    """Run the Greedy Semi-Soft GoDec Algorithm (GreBsmo).

    Parameters
    ----------
    D : array
        nxp data matrix with n samples and p features
    ranks : list of int
        rank(L)<=rank
    tau : float
        soft thresholding
    inpower : float
        >=0, power scheme modification, increasing it lead to better accuracy and more time cost
    k : int
        rank stepsize

    Returns
    -------
    L
        Low-rank part
    S
        Sparse part
    RMSE
        error
    error
        ||X-L-S||/||X||

    Notes
    -----
    From GreGoDec.m

    References
    ----------
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix Decomposition in
        Noisy Case", ICML 2011
    Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and Smoothing", AISTATS 2013.

    Tianyi Zhou, 2013, All rights reserved.
    """
    # set rankmax and sampling dictionary
    rankmax = max(ranks)
    outdict = {}
    rks2sam = [int(np.round(rk)) for rk in (np.array(ranks) / k)]
    rks2sam = sorted(rks2sam)

    # matrix size
    m, n = D.shape
    # ok
    if m < n:
        D = D.T
        # ok

    # To match MATLAB's norm on a matrix, you need an order of 2.
    normD = np.linalg.norm(D, ord=2)

    # initialization of L and S by discovering a low-rank sparse SVD and recombining
    rankk = int(np.round(rankmax / k))
    # ok
    error = np.zeros(max(rankk * inpower, 1) + 1)
    # ok
    # LGR.info(error)
    X, s, Y = svds(D, k, which="LM")
    # CHECK svds
    s = np.diag(s)

    X = X.dot(s)
    # CHECK dot notation
    L = X.dot(Y)
    # CHECK dot notation
    S = wthresh(D - L, tau)
    # ok
    T = D - L - S
    # ok
    error[0] = np.linalg.norm(T[:]) / normD
    # CHECK np.linalg.norm types
    iii = 1
    stop = False
    alf = 0
    estrank = -1

    # Define some variables that shouldn't be touched before they're updated.
    X1 = Y1 = L1 = S1 = T1 = None

    # tic;
    # for r=1:rankk
    for r in range(1, rankk + 1):  # CHECK iterator range
        # parameters for alf
        rrank = rankmax
        estrank = 1
        rank_min = 1
        rk_jump = 10
        alf = 0
        increment = 1
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50
        if iii == inpower * (r - 2) + 1:
            iii = iii + inpower

        for i_iter in range(inpower + 1):
            LGR.debug(f"r {r}, i_iter {i_iter}, rrank {rrank}, alf {alf}")

            # Update of X
            X = L.dot(Y.T)
            # CHECK dot notation
            # if estrank==1:
            #    qro=qr(X,mode='economic');   #CHECK qr output formats    #stopping here on 1/12
            # 	X = qro[0];
            # 	R = qro[1];
            # else:
            X, R = qr(X, mode="economic")
            # CHECK qr output formats

            # Update of Y
            Y = X.T.dot(L)
            # CHECK dot notation
            L = X.dot(Y)
            # CHECK dot notation

            # Update of S
            T = D - L
            # ok
            S = wthresh(T, tau)
            # ok

            # Error, stopping criteria
            T = T - S
            # ok
            ii = iii + i_iter - 1
            # ok
            # embed()
            error[ii] = np.linalg.norm(T[:]) / normD
            if error[ii] < tol:
                stop = True
                break

            # adjust estrank
            if estrank == 1:
                dR = abs(np.diag(R))
                drops = dR[:-1] / dR[1:]
                # LGR.info(dR.shape)
                dmx = max(drops)
                imx = np.argmax(drops)
                rel_drp = (rankmax - 1) * dmx / (sum(drops) - dmx)

                if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or (
                    itr_rank > maxitr_reduce_rank
                ):
                    rrank = max([imx, np.floor(0.1 * rankmax), rank_min])
                    estrank = 0
                    itr_rank = 0

                    if rrank != rankmax:
                        rankmax = rrank
                        if estrank == 0:
                            alf = 0
                            continue

            # adjust alf
            ratio = error[ii] / error[ii - 1]
            if np.isinf(ratio):
                ratio = 0
            # LGR.info(ii, error, ratio)

            if ratio >= 1.1:
                increment = max(0.1 * alf, 0.1 * increment)
                X = X1
                Y = Y1
                L = L1
                S = S1
                T = T1
                error[ii] = error[ii - 1]
                alf = 0
            elif ratio > 0.7:
                increment = max(increment, 0.25 * alf)
                alf = alf + increment

            # Update of L
            # LGR.info("updating L")
            X1 = X
            Y1 = Y
            L1 = L
            S1 = S
            T1 = T
            # ipdb.set_trace()
            L = L + ((1 + alf) * (T))

            # Add coreset
            if i_iter > 8:
                if np.mean(error[ii - 7 : ii + 1]) / error[ii - 8] > 0.92:
                    iii = ii
                    sf = X.shape[1]
                    if Y.shape[0] - sf >= k:
                        Y = Y[:sf, :]
                    break

        if r in rks2sam:
            L = X.dot(Y)
            if m < n:
                temp_D = D.T
                temp_L = L.T
                temp_T = T.T
            outdict[r * k] = [temp_L, temp_D - temp_L, temp_D - temp_L - temp_T]

        # Coreset
        if not stop and r < rankk:
            v = np.random.randn(k, np.maximum(m, n)).dot(L)
            Y = np.vstack([Y, v])
            # correct this

        # Stop
        if stop:
            break

    error[error == 0] = None

    return outdict


def run_godec_denoising(
    in_file,
    mask,
    out_dir=".",
    prefix="",
    method="greedy",
    ranks=[4],
    norm_mode="vn",
    thresh=0.03,
    drank=2,
    inpower=2,
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

    if thresh is None:
        mu = masked_data.mean(axis=-1)
        thresh = np.median(mu[mu != 0]) * 0.01

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
        thresh_ = temp_data.std() * thresh
        LGR.info(f"Setting threshold to {thresh_}")
    else:
        temp_data = dnorm.copy()
        thresh_ = thresh

    if method == "greedy":
        # GreGoDec
        godec_outputs = greedy_semisoft_godec(
            temp_data,
            ranks=ranks,
            tau=1,
            tol=1e-7,
            inpower=inpower,
            k=drank,
        )
    else:
        for rank in ranks:
            X_L, X_S, X_G = standard_godec(
                temp_data,
                thresh=thresh_,
                rank=rank,
                power=1,
                tol=1e-3,
                max_iter=500,
                random_seed=0,
            )

            godec_outputs[rank] = [X_L, X_S, X_G]

    if wavelet:
        LGR.info("++Inverse wavelet transforming outputs")
        for rank in godec_outputs.keys():
            godec_outputs[rank] = [idwtmat(arr, cal) for arr in godec_outputs[rank]]

    if norm_mode == "dm":
        for rank in godec_outputs.keys():
            godec_outputs[rank][0] = godec_outputs[rank][0] + rmu[:, np.newaxis]
    elif norm_mode == "vn":
        for rank in godec_outputs.keys():
            godec_outputs[rank][0] = (godec_outputs[rank][0] * rstd[:, np.newaxis]) + rmu[
                :, np.newaxis
            ]
            godec_outputs[rank][1] = godec_outputs[rank][1] * rstd[:, np.newaxis]
            godec_outputs[rank][2] = godec_outputs[rank][2] * rstd[:, np.newaxis]

    metadata = {
        "normalization": norm_mode,
        "wavelet": wavelet,
        "ranks": ranks,
        "k": drank,
        "p": inpower,
        "t": thresh,
    }
    metadata_file = os.path.join(out_dir, "dataset_description.json")
    with open(metadata_file, "w") as fo:
        json.dump(metadata, fo, sort_keys=True, indent=4)

    for rank, outputs in godec_outputs.items():
        lowrank_img = unmask(outputs[0].T, mask)
        sparse_img = unmask(outputs[1].T, mask)
        noise_img = unmask(outputs[2].T, mask)

        lowrank_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_lowrankts.nii.gz")
        )
        sparse_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_bold.nii.gz")
        )
        noise_img.to_filename(
            os.path.join(out_dir, f"{prefix}desc-GODEC_rank-{rank}_errorts.nii.gz")
        )
