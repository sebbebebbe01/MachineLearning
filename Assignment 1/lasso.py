"""
Functions for 1st Assignment in FMAN45 regarding Lasso
"""

import numpy as np
from scipy.signal.windows import hann
from numpy.random import permutation


def lasso_ccd(data, reg_matrix, lambda_val, w_old=None):
    """
    Solves the LASSO optimization problem using cyclic coordinate descent.

    Parameters:
    data       - Nx1 data vector
    reg_matrix - NxM regression matrix (ndarray)
    lambda_val - Scalar lambda value
    w_old      - Mx1 lasso estimate for warm-starting the solution (optional)

    Return:
    w_hat      - Mx1 LASSO estimate using cyclic coordinate descent algorithm
    """

    shape_reg = reg_matrix.shape
    if shape_reg[0] != data.shape[0]:
        raise ValueError(
            f"Shapes of data ({data.shape[0]}x{data.shape[1]})"
            f"and regression matrix ({shape_reg[0]}x{shape_reg[1]}) does not match."
        )
    if w_old is None:
        w_old = np.zeros((shape_reg[1], 1))

    nbr_iter = 50
    update_cycle = 5

    w_hat = w_old
    w_ind = (np.abs(w_hat) > lambda_val).astype(np.int32)
    res = data - np.matmul(reg_matrix, w_hat)

    for i in range(nbr_iter):
        # Snippet below is a common way of speeding up the estimation process. Use it if you like.
        # Basically, only the non-zero estimates are updated at every iteration. The zero estimates
        # are only updated every updatecycle number of iterations. Use to your liking. Otherwise use
        # contents of else statement.
        if (i % update_cycle) and i > 1:
            ind_rand_order = permutation(np.where(w_ind)[0])
        else:
            ind_rand_order = permutation(range(shape_reg[1]))

        for idx in ind_rand_order:
            reg_col = None # ... select current regression vector
            res = None # ... put impact of old w(kind) back into the residual.
            w_hat[idx] = None # ... update the lasso estimate at coordinate kind
            res = None # ... remove impact of newly estimated w(kind) from residual.
            w_ind[idx] = None # update whether w_ind at index idx is zero or not.
    return w_hat


def lasso_cv(data, reg_matrix, lambda_vec, nbr_folds):
    """
    Calculates the LASSO solution problem and trains the hyperparameter using cross-validation.

    Parameters:
    data        - Nx1 data vector
    reg_matrix  - NxM regression matrix (ndarray)
    lambda_vec  - Vector grid of possible hyperparameters
    nbr_folds   - Number of folds

    Return:
    w_opt       - Mx1 LASSO estimate for optimal lambda
    lambda_opt  - Optimal lambda value
    rmse_val    - Vector of validation RMSE values for lambdas in grid
    rmse_est    - Vector of estimation RMSE values for lambdas in grid
    """

    se_val = np.zeros((nbr_folds, len(lambda_vec)))
    se_est = np.zeros((nbr_folds, len(lambda_vec)))

    nbr_est = np.floor(reg_matrix.shape[0] / nbr_folds).astype(np.int32)
    rand_idx = None

    for kfold in range(nbr_folds):
        print(f"Fold: {kfold} of {nbr_folds}")
        val_ind = None
        est_ind = None
        if any(np.isin(est_ind, val_ind)):
            raise ValueError("There are overlapping indices in valind and estind")

        w_old = np.zeros((reg_matrix.shape[1], 1))
        for idx, lambda_val in enumerate(lambda_vec):
            print(f"Value: {idx} of {len(lambda_vec)}")
            w_hat = None
            se_val[kfold, idx] = None
            se_est[kfold, idx] = None
            w_old = w_hat

    rmse_val = np.sqrt(np.mean(se_val, axis=0))
    rmse_est = np.sqrt(np.mean(se_est, axis=0))
    lambda_opt = lambda_vec[np.argmin(rmse_val)]
    w_opt = None

    return w_opt, lambda_opt, rmse_val, rmse_est


def multiframe_lasso_cv(data, reg_matrix, lambda_vec, nbr_folds):
    """
    Calculates the LASSO solution for all frames and trains the hyperparameter using
    cross-validation.

    Parameters
    data        - Nx1 data vector
    reg_matrix  - NxM regression matrix (ndarray)
    lambda_vec  - vector grid of possible hyperparameters
    nbr_folds   - number of folds

    Return:
    W_opt       - MxN frames LASSO estimate for optimal lambda
    lambda_opt  - Optimal lambda value
    rmse_val    - Vector of validation MSE values for lambdas in grid
    rmse_est    - Vector of estimation MSE values for lambdas in grid
    """

    N, M = reg_matrix.shape
    nbr_frames = None
    w_opt = np.zeros((M, nbr_frames))
    se_val = np.zeros((nbr_folds, len(lambda_vec)))
    se_est = np.zeros((nbr_folds, len(lambda_vec)))

    nbr_est = np.floor(N / nbr_folds)
    rand_idx = None

    for frame in range(nbr_frames):
        local_data = data[frame * N : (frame + 1) * N]

        for kfold in range(nbr_folds):
            val_ind = None
            est_ind = None
            if any(np.isin(est_ind, val_ind)):
                raise ValueError("There are overlapping indices in valind and estind")

            w_old = np.zeros(M, 1)
            for idx, lambda_val in enumerate(lambda_vec):
                w_hat = None
                se_val[kfold, idx] = None
                se_est[kfold, idx] = None
                w_old = w_hat
    
    rmse_val = np.sqrt(np.mean(se_val, axis=0))
    rmse_est = np.sqrt(np.mean(se_est, axis=0))
    
    lambda_opt = None
    for frame in range(nbr_frames):
        local_data = data[frame * N : (frame + 1) * N]
        w_opt[:, frame] = None

    return w_opt, lambda_opt, rmse_val, rmse_est


def lasso_denoise(Tnoisy, X, lambda_val):
    """
    Denoises the data in Tnoisy using LASSO estimates for hyperparameter lambdaopt.
    Cycles through the frames in Tnoisy, calculates the LASSO estimate, selecting
    the non-zero components and reconstructing the data using these components only,
    using a WOLS estimate, weighted by the Hanning window.

    Parameters:
    - Tnoisy (numpy.ndarray): NNx1 noisy data vector
    - X (numpy.ndarray): NxM regression matrix
    - lambda_val (float): Hyperparameter value (selected from cross-validation)

    Returns:
    - Yclean (numpy.ndarray): NNx1 denoised data vector
    """

    # Sizes
    NN = len(Tnoisy)
    N, M = X.shape

    # Frame indices parameters
    loc = 0
    hop = N // 2
    idx = np.arange(N)

    Z = np.diag(hann(N))  # Weight matrix
    Yclean = np.zeros_like(Tnoisy)  # Clean data preallocation

    while loc + N <= NN:
        t = Tnoisy[loc + idx]  # Pick out data in the current frame
        wlasso = lasso_ccd(t, X, lambda_val)  # Calculate LASSO estimate
        nzidx = np.abs(wlasso.reshape(-1)) > 0  # Find nonzero indices

        # Calculate weighted OLS estimate for nonzero indices
        wols = np.linalg.lstsq(Z @ X[:, nzidx], Z @ t, rcond=None)[0]
        # Reconstruct denoised signal
        Yclean[loc + idx] += Z @ X[:, nzidx] @ wols

        loc += hop  # Move indices for the next frame
        print(f"{int(loc / NN * 100)} %")  # Show progress

    print("100 %")
    return Yclean
