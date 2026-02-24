# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:41:17 2025

@author: pbrock
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

# DataFrame Libraries
import pandas as pd
import sqlite3

# Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Scientific Libraries
import numpy as np

# System
import os
os.chdir(path + "Code/") 
import General_Functions as GF 

#To save the model
import json

#%% Functions

def rff_prepare_data(df, months, target_col):
    """
    Subset the signal DataFrame to a given set of months, sort, and
    compute cross-sectionally demeaned targets.

    For all rows whose 'eom' is in 'months', this function:
      - Sorts by ['eom', 'id'] for stable alignment.
      - Computes the cross-sectional mean of 'target_col' within each 'eom'.
      - Constructs a DataFrame with ['id', 'eom', demeaned_target].

    Parameters
    ----------
    df : pd.DataFrame
        Full panel of signals and target. Must contain columns
        ['eom', 'id', target_col] and all feature columns.
    months : iterable
        Collection of month end dates (same type as df['eom']) to keep.
    target_col : str, default 'tr_m_sp500_ld1'
        Name of the target return column to demean within each cross-section.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix for the selected months, sorted by ['eom', 'id'],
        with 'target_col' removed.
    y_cs : pd.DataFrame
        DataFrame with columns:
          - 'id'
          - 'eom'
          - f"{target_col}_cs_demeaned": target demeaned within each 'eom'.
        Ordered consistently with 'X'.
    meta : pd.DataFrame
        DataFrame with columns ['id', 'eom'] for the same rows as 'X' and 'y_cs'.
    """
    mask = df['eom'].isin(months)
    X = df.loc[mask].sort_values(['eom', 'id']).copy()

    # meta info for later
    meta = X[['id', 'eom']].copy()

    # cross-sectional demeaned target
    y_cs = pd.concat([meta.reset_index(drop=True),X[target_col].reset_index(drop=True)],
                     axis=1 
                     )

    return X.drop(columns = [target_col]), y_cs, meta

def rff_weights(cont_feat_names, p_max, seed):
    """
    Sample base frequency weights (omegas) for Random Fourier Features.

    The returned matrix contains standard normal draws and is later
    rescaled by the kernel bandwidth 'sigma' and truncated to the
    desired number of random features in 'rff_feature_map'.

    Parameters
    ----------
    cont_feat_names : list of str
        Names of continuous feature columns. Only the length of this list
        is used to set the number of rows of the weight matrix.
    p_max : int
        Maximum total number of RFF features to support. Only half of this
        ('p_max // 2') base frequency vectors are sampled.
    seed : int
        Random seed for reproducible frequency sampling.

    Returns
    -------
    base_W : np.ndarray of shape (n_cont_features, p_max // 2)
        Matrix of base random frequencies (standard normal draws) used
        in the RFF transform.
    """
    rng = np.random.default_rng(seed)
    # Base random weights
    base_W = rng.normal(loc=0.0, scale=1.0, size=(len(cont_feat_names), p_max // 2))
    
    return base_W


def rff_feature_map(X, cont_feat_names, oneHot_feat_names, base_W:np.array, sigma:float, p:int):
    """
    Map input features to Random Fourier Features (RFFs) with per-month standardisation.

    This function:
      1. Selects continuous features 'cont_feat_names' from 'X'.
      2. Projects them onto random frequencies W = base_W / sigma (truncated to p/2 columns).
      3. Applies cosine and sine transforms and scales by sqrt(2 / p).
      4. Standardises the resulting RFFs cross-sectionally within each 'eom'
         (zero mean, unit variance per month).
      5. Optionally appends one-hot / categorical columns 'oneHot_feat_names'
         without further transformation.

    Parameters
    ----------
    X : pd.DataFrame
        Input data containing at least:
          - Continuous feature columns listed in 'cont_feat_names'.
          - Optional one-hot / categorical columns in 'oneHot_feat_names'.
          - An 'eom' column used to define cross-sectional groups.
    cont_feat_names : list of str
        Names of continuous feature columns to be transformed into RFFs.
    oneHot_feat_names : list of str
        Names of columns to be appended as-is (e.g., dummy variables).
        Can be empty.
    base_W : np.ndarray, shape (n_cont_features, p_max // 2)
        Base frequency matrix returned by 'rff_weights'.
    sigma : float
        RBF kernel bandwidth. Frequencies are scaled by 1 / sigma.
    p : int
        Total number of RFF features (must be even). Internally, p/2
        cosine and p/2 sine features are constructed.

    Returns
    -------
    Z : np.ndarray of shape (n_samples, p + n_onehot)
        Design matrix of standardised RFF features (p columns) concatenated
        with any one-hot / categorical features selected by 'oneHot_feat_names'.
    """
    
    # ---------------------------------------------------------
    # 1. Setup and Raw RFF Projection
    # ---------------------------------------------------------
    # Scale weights
    # Shape: (n_features, p//2)
    W = (base_W / sigma)[:, :p//2]
    
    # Projection: X @ W
    proj = X[cont_feat_names].to_numpy() @ W
    
    # Apply non-linearity (Cos and Sin)
    # Shape: (n_samples, p)
    # We create this separately so we can standardise it BEFORE adding categoricals
    Z_cont = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2 / p)

    # ---------------------------------------------------------
    # 2. Efficient Cross-Sectional Standardisation
    # ---------------------------------------------------------
    # We iterate over the groups defined by 'eom' to calculate mean/std per month.
    # X.groupby('eom').indices returns a dict: { '2004-01-31': [0, 1, 2...], ... }
    # These indices align perfectly with Z_cont because X and Z_cont have the same row order.
    
    for _, idx in X.groupby('eom').indices.items():
        # 1. Slice the batch for this specific month
        batch = Z_cont[idx]
        
        # 2. Compute Mean and Std for this month
        mu = np.mean(batch, axis=0)
        std = np.std(batch, axis=0, ddof = 0)
        
        # 3. Handle division by zero (constant features)
        std[std == 0] = 1.0 
        
        # 4. Update the array in-place
        Z_cont[idx] = (batch - mu) / std

    # ---------------------------------------------------------
    # 3. Final Assembly
    # ---------------------------------------------------------
    # Add the categorical features (untouched)
    # Assuming cat_feat_names cols are numerical/dummies. 
    # If they are strings, this will cause type issues.
    
    if len(oneHot_feat_names) > 0:
        Z = np.hstack([Z_cont, X[oneHot_feat_names].to_numpy()])
    else:
        Z = Z_cont

    return Z

def rff_predict(Z, beta):
    """
    Compute predictions from a linear model given RFF features and coefficients.

    This function assumes 'beta' corresponds exactly to the columns of 'Z'
    (i.e., no intercept is added inside this function).

    Parameters
    ----------
    Z : np.ndarray of shape (n_samples, n_features)
        Design matrix of RFF (and possibly additional) features.
    beta : np.ndarray of shape (n_features,)
        Coefficient vector of the fitted linear model.

    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values Z @ beta.
    """
    
    #Compute the prediction
    y_pred = Z @ beta 
    
    return y_pred

def oos_r2_score(y_true, y_pred):
    """
    Compute out-of-sample R-squared (OOS R²) given realised and predicted values.

    The statistic compares the mean squared prediction error to the variance
    of the realised values. It can be negative if the model performs worse
    than a constant-mean benchmark.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Realised (true) target values.
    y_pred : np.ndarray of shape (n_samples,)
        Model predictions for the same observations.

    Returns
    -------
    r2_oos : float
        Out-of-sample R-squared defined as:
            1 - MSE(y_true, y_pred) / Var(y_true)
        Returns 0.0 if Var(y_true) is zero.
    """
    
    mse = np.mean((y_true - y_pred)**2)
    var = np.var(y_true, ddof = 0)
    if var == 0: return 0.0
    return 1 - (mse / var)

def rff_fit_beta(X, y, cont_feat_names, oneHot_feat_names, base_W, params):
    """
    Fit a ridge regression model on RFF-transformed features.

    This is a convenience wrapper that:
      1. Transforms 'X' into RFF (plus optional one-hot) features using
         'rff_feature_map' with the hyperparameters in 'params'.
      2. Solves a ridge-penalised linear regression using either the
         primal or dual formulation depending on the sample/feature ratio.

    No intercept term is added; the model is purely linear in the features
    produced by 'rff_feature_map'. This is appropriate when inputs and/or
    targets are already standardised or demeaned.

    Parameters
    ----------
    X : pd.DataFrame
        Input features for the estimation window. Must contain:
          - Continuous columns in 'cont_feat_names'
          - Optional columns in 'oneHot_feat_names'
          - An 'eom' column used for standardisation in 'rff_feature_map'.
    y : array-like of shape (n_samples,)
        Target values (typically cross-sectionally demeaned returns).
    cont_feat_names : list of str
        Names of continuous feature columns to be transformed into RFFs.
    oneHot_feat_names : list of str
        Names of one-hot / categorical columns to append unchanged.
    base_W : np.ndarray
        Base frequency matrix returned by 'rff_weights'.
    params : dict
        Hyperparameters for the RFF + ridge model with keys:
          - 'sigma'   : float, kernel bandwidth.
          - 'p'       : int, total number of RFF features (even).
          - 'penalty' : float, ridge penalty parameter (lambda).

    Returns
    -------
    beta : np.ndarray of shape (n_features,)
        Ridge regression coefficients corresponding to the columns of the
        transformed design matrix Z returned by 'rff_feature_map'.
    """
    
    #Scale Features
    Z_cs_scaled = rff_feature_map(X, cont_feat_names, oneHot_feat_names, base_W, params['sigma'], params['p'])
    
    n, p = Z_cs_scaled.shape
    penalty = params['penalty']
    
    # 1. Compute ZZ (Primal or Dual)
    if n >= p: # Primal form (Features <= Samples)
        ZZ = Z_cs_scaled.T @ Z_cs_scaled
        ridge_matrix = ZZ + penalty * np.eye(p)
        beta = np.linalg.solve(ridge_matrix, Z_cs_scaled.T @ y)
    else: # Dual form (Samples < Features)
        K = Z_cs_scaled @ Z_cs_scaled.T
        ridge_matrix = K + penalty * np.eye(n)
        alpha = np.linalg.solve(ridge_matrix, y)
        beta = Z_cs_scaled.T @ alpha
    
    return beta

def rff_predict_oos(X, cont_feat_names, oneHot_feat_names, base_W, params, beta):
    """
    Generate out-of-sample predictions using a fitted RFF + ridge model.

    This function applies the same RFF transformation used during training
    (same 'base_W', 'sigma', and 'p') and then uses the supplied coefficient
    vector 'beta' to compute linear predictions.

    Parameters
    ----------
    X : pd.DataFrame
        Out-of-sample feature data. Must contain:
          - Continuous columns in 'cont_feat_names'
          - Optional columns in 'oneHot_feat_names'
          - An 'eom' column for cross-sectional standardisation in
            'rff_feature_map'.
    cont_feat_names : list of str
        Names of continuous feature columns to transform into RFFs.
    oneHot_feat_names : list of str
        Names of one-hot / categorical columns to append unchanged.
    base_W : np.ndarray
        Base frequency matrix returned by 'rff_weights', using the same
        seed and dimensionality as in training.
    params : dict
        Hyperparameters used when fitting the model. Must contain:
          - 'sigma' : float, kernel bandwidth.
          - 'p'     : int, total number of RFF features (even).
    beta : np.ndarray of shape (n_features,)
        Coefficient vector returned by 'rff_fit_beta'.

    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values for the rows in 'X'.
    """
    Z = rff_feature_map(X, cont_feat_names, oneHot_feat_names, base_W, params['sigma'], params['p'])
    
    return rff_predict(Z, beta)

def save_model(filepath, date, params, beta):
    """
    Save a single fitted model configuration and its coefficients in a JSON file.
    No appending is performed. Each call overwrites or creates exactly one file.

    The JSON file contains:
      - "date"    : string, YYYY-MM-DD
      - "sigma"   : float, RFF bandwidth
      - "p"       : int, number of RFF features
      - "penalty" : float, ridge penalty parameter
      - "beta"    : list of floats, regression coefficients

    Parameters
    ----------
    filepath : str or path-like
        Base file path (without extension). A '.json' extension is added.
    date : pd.Timestamp or datetime-like
        Model date. Only the date component is stored.
    params : dict
        Hyperparameter dictionary with keys:
          - 'sigma'   : float
          - 'p'       : int
          - 'penalty' : float
    beta : np.ndarray
        Coefficient vector of the fitted model.
    """

    record = {
        "date": str(date.date()),
        "sigma": float(params['sigma']),
        "p": int(params['p']),
        "penalty": float(params['penalty']),
        "beta": beta.tolist()
    }

    # Save as a normal JSON file, overwriting if it already exists
    fullpath = filepath + ".json"
    with open(fullpath, "w") as f:
        json.dump(record, f, indent=2)

#%% Preliminaries & Data

#=================================================
#               Preliminaries
#=================================================

#Settings
settings = GF.get_settings()
signals = GF.get_signals()

cont_feat_names, oneHot_feat_names = signals[0], signals[1]

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_processed.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")


#=================================================
#               Read in Data
#=================================================

#Target
target_col = 'tr_m_sp500_ld1'

#Load Data
df, signal_months, trading_month_start, feat_cols, \
    window_size, validation_size  \
        = GF.load_retpred_data(db_signals = JKP_Factors,       #Database with signals
                               db_sp500 = SP500_Constituents,  #Database with S&P500 Constituents
                               settings = settings,            #General settings
                               target = target_col,            #Prediction target
                               rank_signals = False,           #Use Zscores
                               trade_start = '2004-01-31',     #First trading date
                               trade_end = '2024-12-31',       #Last trading date
                               continuous_signals = signals[0],#List of continuous signals
                               fill_missing_values = True,     #Fill missing values 
                               CRSP = False,                   #Estimation on S&P 500 universe
                               )

#=================================================
#           Rolling Window Parameters
#=================================================

#Window Size and Validation Periods for rolling window
window_size = settings['rolling_window']['window_size']
validation_size = settings['rolling_window']['validation_periods'] 
test_size = settings['rolling_window']['test_size'] #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

#Trading Dates
trading_dates = signal_months[trading_month_start:]

#Seed
seed = 2025

# ================================================
#        Model Type (Requires Manual Naming)
# ================================================
model_name = "RFF"
target_type = "LevelTrMsp500Target"
file_end = f"SP500UniverseFL_ZscoreFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"
prediction_name = "ret_pred"

#%% RFF Initialisation

# Pre-generate Base Weights (Fixed seed for consistency across time)
# We need the max p to generate a large enough weight matrix once

base_W_global = rff_weights(cont_feat_names, settings['RFF']['p_vec'].max(), seed=seed)

# Container for predictions
predictions = []

# Initialise Best Parameters
best_params = None
best_r2 = -np.inf

months_since_tune = test_size   # start so that the first iteration triggers tuning

#%% Rolling Window Estimation with Hyperparameter Tuning


#=================================================
#               Main Rolling Loop
#=================================================

for trade_idx, date in enumerate(trading_dates, start=trading_month_start):

    #========================================
    # Get the Data for the Rolling Regression
    #========================================
    
    #Rolling Window Dates
    start_idx = max(0, trade_idx - (window_size + validation_size+1))
    window_months = signal_months[start_idx:trade_idx-1]
    #Note: For fitting beta we can only use signals up to t-2 (trading date is date t)
    #       since we are predicting next period's target. 
    #       Else-wise: Look-ahead bias (data leakage)
    
    # Split into train / val months
    train_months = window_months[:window_size]
    val_months   = window_months[window_size:]
    
    # Training Data
    X_train, y_train_cs, _ = rff_prepare_data(df, train_months, target_col)
    
    #Validation Data
    X_val,   y_val_cs, _ = rff_prepare_data(df, val_months, target_col)
    
    # Check if we need to Tune (Once a year or first run)
    if months_since_tune >= test_size:
        
        #========================================
        #           Hyperparameter Tuning
        #========================================
        
        print(f"Re-tuning hyperparameters at date {date}...")
        
        #Initialise objects
        best_r2 = -np.inf
        best_params = None
        
        # Convert to numpy
        y_train_cs = y_train_cs[target_col].to_numpy()
        y_val_cs = y_val_cs[target_col].to_numpy()
    
        for sigma in settings['RFF']['sigma_vec']:
            for p in settings['RFF']['p_vec']:
                
                # Compute Training Features
                Z_train_cs_scaled = rff_feature_map(X_train, cont_feat_names, oneHot_feat_names, base_W_global, sigma, p)
                
                # Compute Validation Features
                Z_val_cs_scaled   = rff_feature_map(X_val, cont_feat_names, oneHot_feat_names, base_W_global, sigma, p)
                
                #-------------------------------------------------------------- 
                #---- Precompute Matrix Products for Ridge (Primal vs Dual) ---
                n, n_feat = Z_train_cs_scaled.shape
                
                # Pre-calculate the core matrix to avoid doing it inside penalty loop
                if n >= n_feat:
                    # Primal
                    Core_Matrix = Z_train_cs_scaled.T @ Z_train_cs_scaled
                    XY_corr = Z_train_cs_scaled.T @ y_train_cs
                    is_primal = True
                else:
                    # Dual
                    Core_Matrix = Z_train_cs_scaled @ Z_train_cs_scaled.T
                    is_primal = False
                #--------------------------------------------------------------
                
                #Loop over penalty which re-uses ZZ.T or Z.TZ (saves many computations)
                for penalty in settings['RFF']['penalty_vec']:
                    # Fast Solve
                    if is_primal:
                        ridge_matrix = Core_Matrix + penalty * np.eye(n_feat)
                        beta = np.linalg.solve(ridge_matrix, XY_corr)
                    else:
                        ridge_matrix = Core_Matrix + penalty * np.eye(n)
                        alpha = np.linalg.solve(ridge_matrix, y_train_cs)
                        beta = Z_train_cs_scaled.T @ alpha
                    
                    # Predict
                    y_val_pred = rff_predict(Z_val_cs_scaled, beta)
                    current_r2 = oos_r2_score(y_val_cs, y_val_pred)
    
                    if current_r2 > best_r2:
                        best_r2 = current_r2
                        best_params = {'sigma': sigma, 'p': p, 'penalty': penalty}
        
        #Saved best_params
        saved_best_params = best_params.copy()
        
        # reset counter
        months_since_tune = 0  
        
        print(f"Best Params selected. sigma = {best_params['sigma']}, "
              f"p = {best_params['p']}, penalty = {best_params['penalty']}, "
              f"R2: {best_r2:.8f}")
        
    else:
        # Reuse previously tuned hyperparameters
        best_params = saved_best_params.copy()
        print(f"Skipping hyperparameter tuning at date {date}, reusing saved params.")
    
    # Increment months since last tune
    months_since_tune += 1
    
    
    #===========================================================
    #           Refit on Train & Validation Data
    #===========================================================
    
    #Get Train + Val Data
    X_window, y_window_cs, window_meta = rff_prepare_data(df, window_months,
                                                    target_col)
    
    #Compute beta
    beta = rff_fit_beta(X_window, y_window_cs[target_col].to_numpy(),
                        cont_feat_names, oneHot_feat_names, base_W_global, best_params)
        
    
    #===========================================================
    #           Predict next month's OOS Return
    #===========================================================
    
    # Make the next month prediction for the out-of-sample period 
    test_mask = (df['eom'] == date - pd.offsets.MonthEnd(1))  # or == date, after you verify
    X_test = df.loc[test_mask].sort_values(by = ['eom', 'id'])
    ids_test = X_test[['id', 'eom']]
    
    y_test_pred = rff_predict_oos(X_test, cont_feat_names, oneHot_feat_names, base_W_global, best_params, beta)
    
    
    #===========================================================
    #                   Save results
    #===========================================================
    
    #At 'eom', predict return for 'eom'+1
    pred_df = ids_test.copy()
    pred_df['ret_pred'] = y_test_pred
    predictions.append(pred_df)
    
    #Save Model
    save_model(path + f"Models/RFF/RFF_{file_end}_date_{date.strftime('%Y%m%d')}", date, best_params, beta)


#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(name = f"{model_name}_{target_type}_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()
