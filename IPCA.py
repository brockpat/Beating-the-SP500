# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 09:57:31 2025

@author: patri

Implements the unrestricted weighted least squares algorithm of IPCA
"""

#%% Libraries

import numpy as np
import pandas as pd
import sqlite3

from scipy.linalg import svd, lstsq

import pickle

path = "C:/Users/patri/Desktop/ML/"

import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Functions

# Managed Portfolios
def managed_portfolios(df, signals, target, float_type):
    """
    Construct IPCA “managed portfolio” for each cross-section.
    
    For each date t (end-of-month), this function forms the characteristic matrix
    Z_t (and automatically adds a constant) and computes:

    - x_{t+1} = (1/N_t) Z_t' r_{t+1}      (L-vector)
    - W_t     = (1/N_t) Z_t' Z_t         (L x L matrix)
    - N_t     = number of assets in the cross-section
    
    Parameters
    ----------
    df : pandas.DataFrame
        Panel data containing at least columns:
        - 'eom' : time identifier (end-of-month)
        - 'id'  : asset identifier
        - 'signals' columns : stock characteristics at time t
        - 'target' column   : realised return r_{i,t+1} aligned with characteristics at t
        If df is not already MultiIndexed by ('eom','id'), the function will set it.
    signals : list[str]
        List of characteristic column names (L-1 characteristics). A constant is added
        internally; do NOT include 'constant' before.
    target : str
        Column name in 'df' holding r_{i,t+1}.
    float_type : numpy dtype (e.g., np.float32, np.float64)
        Numeric dtype used for NumPy arrays in computations.
    
    Returns
    -------
    x_dict : dict
        Dictionary mapping each date to x_{t+1} as a 1D NumPy array of shape (L,).
    W_dict : dict
        Dictionary mapping each date to W_t as a 2D NumPy array of shape (L, L).
    Nt_dict : dict
        Dictionary mapping each date to N_t (int).
    L : int
        Number of characteristics including the constant (L = 1 + len(signals)).
    T : int
        Number of time periods (unique dates).
    dates : pandas.Index
        Sorted unique 'eom' values used in the iteration (length T).
    
    Notes
    -----
    - The constant is automatically prepended as the first column of Z_t.
    - x_{t+1} is returned as a flattened vector (via ravel()).
    - This function assumes 'target' is already aligned as t+1 returns; it does not
      do any shifting/lagging itself.
    """
    
    # Set Index
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['eom', 'id'])
    
    # Extract Date Time Array
    dates = df.index.get_level_values('eom').unique().sort_values()
    
    # Include 'constant' in signals if not included
    if 'constant' in signals:
        print("ERROR: 'constant' already included in signals")
        return 
    
    # Dimensions
    L = 1 + len(signals) # Constant + Signals
    T = len(dates)
    
    # Initialise objects as dictionaries
    x_dict = {}
    W_dict = {}
    Nt_dict = {}

    for date in dates:
        # Slice dataframe to date date
        data_t  = df.xs(date, level='eom')
        N_t     = len(data_t)
        # Get Z_t (N_t x L)
        Z_t     = data_t[signals].values.astype(float_type)
        #  Include the constant
        ones    = np.ones((N_t, 1), dtype=float_type)
        Z_t     = np.column_stack([ones, Z_t])
        # Get r_{t+1} (N_t x 1)
        r_tp1   = data_t[target].values.reshape(-1, 1).astype(float_type)
        # Store in dictionaries using date as key
        Nt_dict[date]   = N_t
        x_dict[date]    = (Z_t.T @ r_tp1).ravel() / N_t
        W_dict[date]  = (Z_t.T @ Z_t) / N_t
        
    return x_dict, W_dict, Nt_dict, L, T, dates

def initialise_IPCA(
    x_mat: np.ndarray,
    L: int,
    K: int,
    float_type,
):
    """
    Initialise IPCA parameters (Gamma_alpha, Gamma_beta) and latent factors (F, F_tilde)
    using the PCA/SVD solution on the managed portfolios matrix.

    Parameters
    ----------
    x_mat : ndarray, shape (L, T)
        Managed portfolios matrix X = [x_2, ..., x_{T+1}] with L characteristics (incl. constant)
        and T time periods.
    L : int
        Number of characteristics including the constant.
    K : int
        Number of latent factors.
    float_type : numpy dtype, default np.float64
        Dtype for returned arrays.

    Returns
    -------
    Gamma_alpha : ndarray, shape (L,)
        Initialised intercept mapping (all zeros).
    Gamma_beta : ndarray, shape (L, K)
        Initialised loading mapping (first K left singular vectors of x_mat).
    F : ndarray, shape (K, T)
        Initialised factors (SVD scores diag(s) @ V').
    F_tilde : ndarray, shape (K+1, T)
        Stacked factors with first row ones: [1; F].
    projection : ndarray, shape (L, T)
        Initial projection: Gamma_alpha[:,None] + Gamma_beta @ F.
    """
    # Γ_α initialised to zero
    Gamma_alpha = np.zeros(L, dtype=float_type)

    # Γ_β initialised as PCA solution (SVD on x_mat)
    U, s, VT = svd(x_mat, full_matrices=False)
    Gamma_beta = U[:, :K].copy().astype(float_type)

    # Factors: f_{t+1} = diag(s) @ V'
    F = (np.diag(s[:K]) @ VT[:K, :]).copy().astype(float_type)  # (K x T)

    # Stacked factors: [1; f]
    T = F.shape[1]
    F_tilde = np.vstack([np.ones((1, T), dtype=float_type).reshape(1,-1), F])  # ((K+1) x T)

    # Initial projection
    projection = Gamma_alpha[:, np.newaxis] + (Gamma_beta @ F_tilde[1:, :])  # (L x T)

    return Gamma_alpha, Gamma_beta, F, F_tilde, projection

def factors_and_loadings_IPCA(
    x_mat: np.ndarray,
    W_dict: dict,
    dates,
    Gamma_alpha: np.ndarray,
    Gamma_beta: np.ndarray,
    F: np.ndarray,
    L: int,
    K: int,
    float_type,
    regularise = False
):
    """
    One ALS step for IPCA:
      (1) Update factors F given (Gamma_alpha, Gamma_beta)
      (2) Update (Gamma_alpha, Gamma_beta) given F via vec(Gamma_tilde) least squares.

    Parameters
    ----------
    x_mat : ndarray, shape (L, T)
        Managed portfolios matrix with columns x_{t+1}.
    W_dict : dict[date -> ndarray], each shape (L, L)
        Cross-sectional second moment matrices W_t = (1/N) Z_t' Z_t.
    dates : iterable
        Ordered dates corresponding to columns of x_mat and keys of W_dict.
    Gamma_alpha : ndarray, shape (L,)
        Current intercept mapping.
    Gamma_beta : ndarray, shape (L, K)
        Current loading mapping.
    F : ndarray, shape (K, T)
        Current factor matrix (will be updated and returned).
    L : int
        Number of characteristics (incl. constant).
    K : int
        Number of factors.
    float_type : numpy dtype, default np.float64
        Dtype for internal arrays in the vec(Gamma_tilde) step.

    Returns
    -------
    Gamma_alpha_new : ndarray, shape (L,)
        Updated Gamma_alpha.
    Gamma_beta_new : ndarray, shape (L, K)
        Updated Gamma_beta.
    F_new : ndarray, shape (K, T)
        Updated factors.
    F_tilde_new : ndarray, shape (K+1, T)
        Updated stacked factors [1; F_new].
    """
    T = x_mat.shape[1]

    # ==================
    # 1) Update f_{t+1}
    # ==================
    for t, date in enumerate(dates):
        x_tp1 = x_mat[:, t]
        W_t = W_dict[date]

        rhs = Gamma_beta.T @ (x_tp1 - W_t @ Gamma_alpha)      # (K,)
        lhs = Gamma_beta.T @ W_t @ Gamma_beta                # (K x K)
        
        if regularise:
            lam = 1e-6 * (np.trace(lhs) / lhs.shape[0])
            lhs +=  lam * np.eye(lhs.shape[0], dtype = float_type)

        # Solve lhs f = rhs via least squares (SVD-based)
        F[:, t] = lstsq(lhs, rhs)[0]

    # Update F_tilde
    F_tilde = np.vstack([np.ones((1, T), dtype=float_type), F])

    # ======================
    # 2) Update Γ_β and Γ_α
    # ======================
    # Compute vec(Γ~) without explicitly computing the inverse. Initialise:
    rhs = np.zeros(L * (K + 1), dtype=float_type)
    lhs = np.zeros((L * (K + 1), L * (K + 1)), dtype=float_type)

    # Compute lhs and rhs
    for t, date in enumerate(dates):
        tilde_f_tp1 = F_tilde[:, t]   # (K+1,)
        x_tp1 = x_mat[:, t]           # (L,)
        W_t = W_dict[date]            # (L x L)

        lhs += np.kron(np.outer(tilde_f_tp1, tilde_f_tp1), W_t)  # ((K+1)L x (K+1)L)
        rhs += np.kron(tilde_f_tp1, x_tp1)                       # ((K+1)L,)
    
    if regularise:
        lam = 1e-6 * (np.trace(lhs) / lhs.shape[0])
        lhs +=  lam * np.eye(lhs.shape[0], dtype = float_type)
    
    # Solve for vec(Γ~)
    vec_Gamma = lstsq(lhs, rhs)[0]  # ((K+1)L,)
    
    # Invert column-wise vec operator: first L elements → 1st column, next L element 2nd column, ...
    G_tilde = vec_Gamma.reshape((L, K + 1), order="F")  # [Gamma_alpha, Gamma_beta]

    Gamma_alpha_new = G_tilde[:, 0].copy()
    Gamma_beta_new = G_tilde[:, 1:].copy()

    return Gamma_alpha_new, Gamma_beta_new, F, F_tilde

def identification_IPCA(
    Gamma_alpha: np.ndarray,
    Gamma_beta: np.ndarray,
    F: np.ndarray,
    T: int,
    K: int,
    float_type,
):
    """
    Apply IPCA identification / normalisation steps:

      (a) Orthogonalise Gamma_beta (SVD) and rotate factors accordingly.
      (b) Diagonalise and order factor second moments (eigendecomposition) via rotation.
      (c) Orthogonalise Gamma_alpha w.r.t Gamma_beta via xi-shift and adjust factors.
      (d) Enforce sign convention: each factor time-series mean is positive.

    Parameters
    ----------
    Gamma_alpha : ndarray, shape (L,)
        Current intercept mapping.
    Gamma_beta : ndarray, shape (L, K)
        Current loading mapping.
    F : ndarray, shape (K, T)
        Current factors.
    T : int
        Number of time periods.
    K : int
        Number of factors.
    float_type : numpy dtype, default np.float64
        Dtype for returned arrays.

    Returns
    -------
    Gamma_alpha_id : ndarray, shape (L,)
        Identified/normalised Gamma_alpha.
    Gamma_beta_id : ndarray, shape (L, K)
        Identified/normalised Gamma_beta.
    F : ndarray, shape (K, T)
        Identified/normalised factors.
    F_tilde : ndarray, shape (K+1, T)
        Stacked identified factors [1; F_id].
    """
    
    # --- (a) Orthogonalise Γ_β via SVD and rotate factors
    # SVD
    U, Svals, Vt = svd(Gamma_beta, full_matrices=False)
    # Rotate Γ_β
    Gamma_beta = U[:, :K]
    # Rotate every f_{t+1}
    R = (np.diag(Svals[:K]) @ Vt[:K, :]).astype(float_type)  # (K x K)
    F = (R @ F)

    # --- (b) Diagonalise factor 2nd moment matrix and order descending
    Sigma_f = (F @ F.T) / T # Estimate 2nd moment matrix
    evals, evecs = np.linalg.eigh(Sigma_f)  # Eigenvalues and Eigenvectors
    idx = np.argsort(evals)[::-1]           # Descending order
    V = evecs[:, idx].astype(float_type)    # Eigenvector matrix
    
    # Rotate factors and Γ_β (Γ_β'Γ_β = I kept in tact)
    F = (V.T @ F)
    Gamma_beta = (Gamma_beta @ V)

    # --- (c) Orthogonalise Γ_α against Γ_β and adjust factors (xi shift)
    # OLS
    xi = lstsq(Gamma_beta.T @ Gamma_beta, Gamma_beta.T @ Gamma_alpha)[0]  # (K,)
    # Orthogonalise 
    Gamma_alpha = (Gamma_alpha - Gamma_beta @ xi)
    F = (F + xi.reshape(-1, 1))

    # --- (d) Sign restriction: make factor means positive
    # Compute mean of each f_k (row-wise mean of F)
    f_mean = F.mean(axis=1)
    
    # Flip signs where necessary
    for k in range(K):
        if f_mean[k] < 0:
            F[k, :] *= -1.0
            Gamma_beta[:, k] *= -1.0

    F_tilde = np.vstack([np.ones((1, T), dtype=float_type), F])

    return Gamma_alpha, Gamma_beta, F, F_tilde

def convergence_IPCA(
    Gamma_alpha: np.ndarray,
    Gamma_beta: np.ndarray,
    F_tilde: np.ndarray,
    projection_old: np.ndarray,
    Gamma_alpha_old: np.ndarray,
    Gamma_beta_old: np.ndarray,
    F_tilde_old: np.ndarray,
    mode: str = "projection"
):
    """
    Compute IPCA convergence error with two selectable modes:

    1) mode="projection" (default):
        err = max_{l,t} { projection_new(l,t) - projection_old(l,t) }
        where projection_new = Gamma_alpha[:,None] + Gamma_beta @ F

    2) mode="params":
        err = max(tol_g, tol_f)
        tol_g = max( max|Gamma_beta-Gamma_beta_old|, max|Gamma_alpha-Gamma_alpha_old| )
        tol_f = max|F - F_old|

    Parameters
    ----------
    Gamma_alpha : ndarray, shape (L,)
        Current Gamma_alpha.
    Gamma_beta : ndarray, shape (L, K)
        Current Gamma_beta.
    F_tilde : ndarray, shape (K+1, T)
        Current factors (required if mode="projection").
    projection_old : ndarray, shape (L, T)
        Previous projection matrix (required if mode="projection").
    Gamma_alpha_old : ndarray
        Previous Gamma_alpha (required if mode="params").
    Gamma_beta_old : ndarray
        Previous Gamma_beta (required if mode="params").
    F_old : ndarray
        Previous factors (required if mode="params").
    mode : {"projection","params"}
        Which convergence metric to compute.

    Returns
    -------
    projection_new : ndarray, shape (L, T)
        Current projection matrix.
    err : float
        Convergence error according to selected mode.

    Raises
    ------
    ValueError
        If mode incorrectly provided
    """
    projection_new = Gamma_alpha[:, np.newaxis] + (Gamma_beta @ F_tilde[1:,:])

    if mode == "projection":
        err = np.max(np.abs(projection_new - projection_old))
        return projection_new, err

    if mode == "params":
        tol_g = max(
            np.max(np.abs(Gamma_beta - Gamma_beta_old)),
            np.max(np.abs(Gamma_alpha - Gamma_alpha_old)),
        )
        tol_f = np.max(np.abs(F_tilde[1:,:] - F_tilde_old[1:,:]))
        err = max(tol_g, tol_f)
        return projection_new, err

    raise ValueError("mode must be either 'projection' or 'params'.")


def fit_IPCA(df, signals, target, float_type = np.float64, num_factors = 5, max_iter = 300, tol = 1e-4):
    """
    Fit the Instrumented PCA (IPCA) model via alternating (weighted) least squares.

    This implements the Kelly, Pruitt & Su (2019) IPCA estimator in an ALS loop:

      1) Given loadings (Gamma_alpha, Gamma_beta), update factors f_{t+1} by OLS
         using managed portfolios x_{t+1} and characteristic second moments W_t.

      2) Given factors, update the full mapping matrix \\tilde{Gamma} = [Gamma_alpha, Gamma_beta]
         by solving a linear system for vec(\\tilde{Gamma}).

      3) Enforce identification/normalisation:
         (a) Orthogonalise Gamma_beta (SVD) and rotate factors accordingly.
         (b) Diagonalise and order factor second moments (eigendecomposition) via rotation.
         (c) Orthogonalise Gamma_alpha w.r.t. Gamma_beta via an OLS residualisation (xi-shift).
         (d) Enforce sign convention: each factor has positive time-series mean.

    The algorithm stops when the supremum norm of the change in the “projection”
    (Gamma_alpha + Gamma_beta f_t) falls below 'tol', or when 'max_iter' is reached.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel data containing at least columns:
        - 'eom' : time identifier (end-of-month)
        - 'id'  : asset identifier
        - 'signals' columns : stock characteristics at time t
        - 'target' column   : realised return r_{i,t+1} aligned with characteristics at t
        If df is not already MultiIndexed by ('eom','id'), the function will set it.
    signals : list[str]
        List of characteristic column names (L-1 characteristics). A constant is added
        internally; do NOT include 'constant' before in this list.
    target : str
        Column name in 'df' holding r_{i,t+1}.
    float_type : numpy dtype, default np.float64
        Numeric dtype used in computations. Lower precision (float32) is faster but can
        make the Kronecker-based \\tilde{Gamma} step less stable (see Notes).
    num_factors : int, default 5
        Number of latent factors K.
    max_iter : int, default 300
        Maximum number of ALS iterations.
    tol : float, default 1e-4
        Convergence tolerance for the supremum norm of the change in the projection
        matrix across iterations.

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'Gamma_alpha'    : ndarray, shape (L,)
            Characteristic-to-intercept mapping (includes constant row).
        - 'Gamma_beta'     : ndarray, shape (L, K)
            Characteristic-to-loading mapping.
        - 'F_tilde'        : ndarray, shape (K+1, T)
            Stacked factors with first row ones: [1; f_{t+1}].
        - 'Training Dates' : pandas.Index
            Sorted dates used for estimation.
        - 'iterations'     : int
            Number of ALS iterations performed.
        - 'Error'          : float
            Final convergence error (supremum norm).
        - 'Flag'           : bool
            True if converged within tolerance, else False.

    Notes
    -----
    - With float64 the Gamma and F matrices converge up to desired tolerances.

    - The float64 type only really matters when computing vec(Γ~). In other words, it is
      sufficient to set the float type to 64-bit for the lhs and rhs when computing vec(Γ~).
      All other float types can remain 32-bit.

    - The regularisation can help in the float64 case, but in the float32 case it is
      rather damaging as the precision is low and adding this noise term is a further
      detriment.

    - Z-scores help the ALS algorithm to be much more stable since the W matrix has a
      diagonal of 1. Rank scores took longer to converge and there were more jumps in
      the algorithm, i.e. sometimes it looked like it was converging, but then jumped to
      higher errors and then converged again.

    - lstsq is more stable than linalg.solve() as linalg.solve() assumes a well-conditioned
      matrix and uses LU decomposition, whereas lstsq uses SVD and drops too small singular
      values.

    Additional implementation notes:

    - Initialisation:
        * Gamma_alpha is initialised to zero.
        * Gamma_beta is initialised using PCA/SVD on the managed portfolios matrix
          X = [x_2, ..., x_{T+1}] (shape L x T), taking the first K left singular vectors.
        * Initial factors are taken from the SVD scores (diag(s) @ V').

    - This implementation uses 'numpy.linalg.lstsq' throughout (instead of explicit inverses
      or 'np.linalg.solve') for numerical robustness.
    """
    
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['eom', 'id'])
    
    # Get Managed Portfolio (preliminary computations)
    x_dict, W_dict, Nt_dict, L, T, dates = managed_portfolios(df, signals, target, float_type)
    
    # Unpack Arguments
    K = num_factors
    x_mat = np.column_stack([x_dict[t] for t in dates]) # (L x T). Each column is x_{t+1}
    
    # ---- Initialise Γ_β and Γ_α ----
    Gamma_alpha, Gamma_beta, F, F_tilde, projection = \
        initialise_IPCA(x_mat, L, K, float_type)
    
    #Initialise Convergence flag
    flag = False
        
    # ===== ALS =====
    for iteration in range(max_iter):
        # Save Old iterations for convergence check
        Gamma_beta_old = Gamma_beta.copy()
        Gamma_alpha_old = Gamma_alpha.copy()
        vec_Gamma_old = np.column_stack([Gamma_alpha_old, Gamma_beta_old]).reshape(-1, order="F")
        F_old = F.copy()
        F_tilde_old = F_tilde.copy()
        projection_old = projection.copy()
        
        # ---- Update factors and Γ~ ----
        Gamma_alpha, Gamma_beta, F, F_tilde = \
            factors_and_loadings_IPCA(x_mat, W_dict, dates, Gamma_alpha,
                                      Gamma_beta, F, L, K, float_type)
            
        # ---- Identification ----
        Gamma_alpha, Gamma_beta, F, F_tilde = \
            identification_IPCA(Gamma_alpha, Gamma_beta, F, T,
                                     K, float_type)
    
        # ---- Check convergence ----
        projection, err = \
            convergence_IPCA(Gamma_alpha, Gamma_beta, F_tilde, 
                                   projection_old, Gamma_alpha_old, 
                                   Gamma_beta_old, F_tilde_old)
        
        # Stop if convergence achieved
        if err < tol:
            print(f"Converged after iteration {iteration}")
            flag = True
            break
        
    return {
        "Gamma_alpha":      Gamma_alpha,
        "Gamma_beta":       Gamma_beta,
        "F_tilde":          F_tilde,
        "Training Dates":   dates,
        "iterations":       iteration + 1,
        "Error":            err,
        "Flag":             flag
        }

def predict_IPCA(df, signals, IPCA_model, prediction_name):
    """
    Generate one-step-ahead IPCA return forecasts for a given cross-section.
    
    This function produces predicted returns using the fitted characteristic mappings
    (Gamma_alpha, Gamma_beta) and an out-of-sample factor forecast. The factor forecast
    used here is a simple baseline: the in-sample time-series mean of the estimated
    factors (i.e., \\hat f_{t+1|t} = mean_t \\hat f_{t+1}).
    
    For a cross-section with characteristics Z_t (including a constant), the forecast is:
        \\hat r_{t+1} = Z_t Gamma_alpha + Z_t Gamma_beta \\bar f
    where \\bar f is the in-sample mean of the estimated latent factors.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cross-sectional data for a single date
        containing at least:
        - 'eom' and 'id' (for indexing in the returned DataFrame)
        - 'signals' columns with characteristics Z_t (WITHOUT a constant column)
    signals : list[str]
        List of characteristic column names used in the fitted model (excluding the constant).
        The function will prepend a constant internally; do NOT include 'constant' before.
    IPCA_model : dict
        Output dictionary from 'fit_IPCA', containing at least:
        - 'Gamma_alpha' : ndarray (L,)
        - 'Gamma_beta'  : ndarray (L, K)
        - 'F_tilde'     : ndarray (K+1, T)
    prediction_name : str
        Name of the prediction column in the returned DataFrame.
    
    Returns
    -------
    prediction : pandas.DataFrame
        DataFrame with a single column 'prediction_name', indexed by ('eom','id'),
        containing predicted returns for each asset in 'df'.
    
    Notes
    -----
    - This function does not estimate factors out-of-sample beyond the mean forecast.
      More sophisticated factor forecasting (e.g., AR models on factors) can be plugged in.
    - 'df' is returned with a MultiIndex ('eom','id'). If 'df' is not already MultiIndexed,
      the function will set it before constructing the output.
    """
    
    # Unpack IPCA model
    Gamma_alpha = IPCA_model['Gamma_alpha']
    Gamma_beta  = IPCA_model['Gamma_beta']
    F           = IPCA_model['F_tilde'][1:,:] #onl get f and not f~
    
    # Forecast f_{t+1} OOS by computing the in-sample mean
    f_mean      = F.mean(axis=1)
    
    # Get Characteristics & No. of Stocks.
    Z_t     = df[signals].values
    N_t     = len(Z_t)
    #  Include the constant in Z_t
    if 'constant' in signals:
        print("ERROR: 'constant' already included in signals")
        return 
    ones    = np.ones((N_t, 1))
    Z_t     = np.column_stack([ones, Z_t])
    
    # Compute the return forecast
    prediction = Z_t @ Gamma_alpha + Z_t @ Gamma_beta @ f_mean
    
    # Create the indexed DataFrame
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['eom', 'id'])
    
    prediction = pd.DataFrame({prediction_name: prediction}, index=df.index)
    
    return prediction

#%% Preliminaries & Data

# =================================================
#               Preliminaries
# =================================================

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_processed.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")

#Get Settings & signals
settings = GF.get_settings()
signals = GF.get_signals()
feat_cols = signals[0] + signals[1] #Continuous and Categorical Features

# =================================================
#                Read in Data
# =================================================

#Target
target_col = 'tr_m_sp500_ld1'

#Load Data
df, signal_months, trading_month_start, feat_cols, \
    window_size, validation_size  \
        = GF.load_retpred_data(db_signals = JKP_Factors,       #Database with signals
                               db_sp500 = SP500_Constituents,  #Database with S&P500 Constituents
                               settings = settings,            #General settings
                               target = target_col,            #Prediction target
                               rank_signals = False,           #Use ZScores
                               trade_start = '2004-01-31',     #First trading date
                               trade_end = '2024-12-31',       #Last trading date
                               continuous_signals = signals[0],#List of continuous signals
                               fill_missing_values = True,     #Fill missing values 
                               CRSP = True,                    #Estimation on CRSP universe or S&P 500 universe
                               )

# ================================================
#           Rolling Window Parameters
# ================================================

#Window Size and Validation Periods for rolling window
window_size = settings['rolling_window']['window_size']
validation_size = settings['rolling_window']['validation_periods'] 
test_size = settings['rolling_window']['test_size'] #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

#Trading Dates
trading_dates = signal_months[trading_month_start:]

# ================================================
#        Model Type (Requires Manual Naming)
# ================================================
model_name = "IPCA"
target_type = "LevelTrMsp500Target"
file_end = f"CRSPUniverse_ZscoreFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"
prediction_name = "ret_pred_IPCA"

#%% IPCA

predictions = []

for trade_idx, date in enumerate(trading_dates, start=trading_month_start):
    
    print(f"Predicting date: {date}")
    
    #========================================
    #               Train IPCA
    #========================================
    
    #Rolling Window Dates
    start_idx = max(0, trade_idx - (window_size + validation_size+1))
    window_months = signal_months[start_idx:trade_idx-1]
    #Note: we can only use signals up to t-2 since we are predicting next period's targelt. Else-wise: Look-ahead bias (data leakage)
    
    #Get Train + Val Data   
    train_df = df[df['eom'].isin(window_months)].copy()
    
    # Fit the Unrestricted IPCA model
    print(f"   Training on {window_months.min()} - {window_months.max()}")
    IPCA_model = fit_IPCA(train_df, feat_cols, target_col)
    
    # Save Model
    with open(path + f"Models/{model_name}_{target_type}_{file_end}_date_{date.strftime('%Y-%m-%d')}.pickle", "wb") as f:
        pickle.dump(IPCA_model, f)
    
    #========================================
    #     Predict next month's OOS Return
    #========================================
    
    test_date = date - pd.offsets.MonthEnd(1)
    test_mask = (df['eom'] == test_date)
    df_test = df.loc[test_mask].copy()
    
    prediction = predict_IPCA(df_test, feat_cols, IPCA_model, prediction_name)
    
    predictions.append(prediction.reset_index())

#%% Save Predictions
df_predictions = pd.concat(predictions)

#At eom, the prediction is for eom+1
df_predictions.to_sql(name = f"{model_name}_{target_type}_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()

