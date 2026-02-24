# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:38:52 2025

@author: Patrick

===============================================================================
OVERVIEW: Estimation of the Conditional Stock Return Covariance Matrix (Σ_t)
===============================================================================

This file implements the Barra-style covariance estimator.
The conditional covariance matrix of stock returns is constructed as

    Σ_t = S_t V(f_t) S_t' + diag(V(ε_t)),

where:
    - S_t         : matrix of factor loadings at time t
    - V(f_t)      : factor covariance matrix
    - V(ε_t)      : idiosyncratic variance vector (diagonal)

Instead of constructing the full N×N covariance matrix directly, the code
estimates and stores its three building blocks separately for efficiency and
numerical stability.

-------------------------------------------------------------------------------
1) FACTOR LOADINGS S_t (Characteristic-Based Risk Exposures)
-------------------------------------------------------------------------------

Step 1: Cross-sectional feature ranking
    - All stock characteristics are converted to percentile ranks within each
      month (cross-sectional ECDF transform).
    - Missing values are filled with 0.5 (neutral rank).
    - This standardizes characteristics across time and scale.

Step 2: Cluster construction
    - Individual characteristics are grouped into thematic clusters.
    - For each cluster:
        • Apply direction adjustment (if necessary).
        • Take the cross-sectional mean of ranked characteristics.
    - Result: One exposure per cluster per stock.

Step 3: Cross-sectional standardization
    - Cluster exposures are z-scored within each month.
    - FF12 industry dummies are included as additional factors.

The final factor loading matrix S_t consists of:
    [FF12 industry dummies | cluster z-score exposures]

These exposures are lagged one month when merged with daily returns to ensure
only observable information is used.

-------------------------------------------------------------------------------
2) DAILY FACTOR RETURN ESTIMATION
-------------------------------------------------------------------------------

For each trading day d, the following cross-sectional regression is estimated:

    r_{i,d} = S_{i,t} f_d + ε_{i,d}

where:
    - r_{i,d} : daily excess return
    - S_{i,t} : month-t factor loadings
    - f_d     : daily factor returns
    - ε_{i,d} : residual (specific return)

Estimation details:
    - OLS solved via (X'X)^{-1}X'y
    - Falls back to pseudo-inverse if singular
    - Stored outputs:
        • Daily factor returns f_d
        • Daily residuals ε_{i,d}

-------------------------------------------------------------------------------
3) FACTOR COVARIANCE MATRIX V(f_t)
-------------------------------------------------------------------------------

For each month-end date t:

    • Use last 2520 daily observations (10 years)
    • Apply exponentially weighted moving average (EWMA)

Half-life parameters:
    - Correlations: 378 trading days
    - Variances:    126 trading days

Estimation procedure:
    1) Estimate weighted correlation matrix using half-life 378.
    2) Estimate weighted variances using half-life 126.
    3) Construct covariance matrix as:

           V(f_t) = D^{1/2} Corr(f_t) D^{1/2}

       where D contains the estimated factor variances.

    4) Convert daily covariance to monthly by multiplying by 21.

The separation of correlation and variance estimation improves numerical
stability and helps maintain positive semi-definiteness.

-------------------------------------------------------------------------------
4) IDIOSYNCRATIC VARIANCE diag(V(ε_t))
-------------------------------------------------------------------------------

For each stock i:

    • Estimate EWMA volatility of daily residuals ε_{i,d}
    • Half-life: 126 trading days
    • Initial variance: first 63 valid observations

Data-quality requirement:
    - At least 200 non-missing residual observations within last 252 days.
    - Otherwise, impute using:
        1) Median residual volatility within same size group
        2) If still missing, overall cross-sectional median

Monthly aggregation:
    - Keep last trading day of each month
    - Convert daily variance to monthly by multiplying by 21

-------------------------------------------------------------------------------
5) FINAL OUTPUT: BARRA COVARIANCE OBJECT
-------------------------------------------------------------------------------

For each month t, the dictionary `barra_cov[t]` stores:

    {
        "fct_load" : S_t              (stock × factor loadings matrix),
        "fct_cov"  : V(f_t)           (factor covariance matrix),
        "ivol_vec" : diag(V(ε_t))     (vector of idiosyncratic variances)
    }

The full covariance matrix can be reconstructed as:

    Σ_t = S_t V(f_t) S_t' + diag(ivol_vec)

This structured representation is computationally efficient and suitable
for portfolio optimization, risk forecasting, and mean-variance allocation.

-------------------------------------------------------------------------------
KEY DESIGN CHOICES
-------------------------------------------------------------------------------

• 10-year rolling window (2520 trading days)
• Separate half-lives for correlations and variances
• EWMA smoothing for both factor and idiosyncratic components
• Cross-sectional normalization of characteristics
• Industry controls (FF12)
• Robust imputation for missing residual volatility

This implementation closely follows the paper’s covariance specification
and provides a stable, economically interpretable risk model suitable for
large cross-sectional equity universes.
===============================================================================
"""


#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

import os
os.chdir(path + "Code/")

import pickle

import pandas as pd
import sqlite3

import numpy as np
from tqdm import tqdm
from numba import njit

#Import user-specific functions
import General_Functions as GF

#%%
# =============================================================================
#                           Preliminaries
# =============================================================================

industries = True
obs = 252*10 #10 years of previous data to estimate Sigma_t
hl_cor = int(252 * 3 / 2) #half life covariances
hl_var = int(252 / 2) #half live variances
hl_stock_var = int(252 / 2)
min_stock_obs = 200
initial_var_obs = 21*3

# Compute weights for exponential moving average
time_range = np.arange(obs, 0, -1)
w_cov = (0.5 ** (1 / hl_cor)) ** time_range
w_var = (0.5 ** (1 / hl_var)) ** time_range

settings = GF.get_settings()

#%% Functions

# Compute Cluster Ranks
def build_cluster_ranks(cluster_data_m, cluster_labels, clusters, features):
    
    # Initialize an empty DataFrame to hold the cluster rank columns
    cluster_ranks = pd.DataFrame(index=cluster_data_m.index)
    
    #Iterate over each cluster
    for cl in clusters:
        # Extract Characteristics belonging to this Cluster
        chars_sub = cluster_labels[(cluster_labels['cluster'] == cl) & 
                                   (cluster_labels['characteristic'].isin(features))]
        
        # Get Data of Characteristics belonging to the cluster
        data_sub = cluster_data_m[chars_sub['characteristic'].tolist()].copy()
        
        # Apply direction transformation
        for c in chars_sub['characteristic']:
            #Get Direction
            dir_val = chars_sub.loc[chars_sub['characteristic'] == c, 'direction'].values[0]
            if dir_val == -1:
                data_sub[c] = 1 - data_sub[c]
        
        # Compute row means (each row is a stock at a time point t) 
        #   and assign to a new column named after the cluster
        cluster_ranks[cl] = data_sub.mean(axis=1)
                
    return cluster_ranks

# Weighted Covariance Matrix
def weighted_cov_wt(df, weights):
    """
    Compute the weighted covariance matrix just like R's cov.wt(..., center=TRUE, method="unbiased").

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of shape (n, p) with numeric data (excluding 'date' or other non-numeric columns).
    weights : array-like
        Length-n vector of weights (e.g. from tail(w_cor, t)). Will be normalized.

    Returns
    -------
    cov_df : pandas.DataFrame
        Weighted covariance matrix as a DataFrame (p x p) with variable names.
    """
    # Convert to NumPy
    X = df.to_numpy()
    w = np.asarray(weights, dtype=float)
    
    # Normalize weights
    w_norm = w / w.sum()
    
    # Weighted mean
    mu = np.average(X, axis=0, weights=w_norm)   
    
    # Centered data
    Xc = X - mu
    
    # Apply sqrt(weights) to each row
    Xw = Xc * np.sqrt(w_norm)[:, np.newaxis]
    
    # Unbiased weighted covariance
    denom = 1.0 - np.sum(w_norm ** 2)
    cov_matrix = (Xw.T @ Xw) / denom  # shape (p, p)

    # Return as DataFrame with labels
    cov_df = pd.DataFrame(cov_matrix, index=df.columns, columns=df.columns)
    
    return cov_df

def weighted_cor_wt(df, weights):
    """
    Compute the weighted correlation matrix like R's cov.wt(..., cor=TRUE, center=TRUE, method="unbiased").

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric data (n x p), excluding any non-numeric columns like 'date'.
    weights : array-like
        Length-n vector of weights (will be normalized).

    Returns
    -------
    cor_df : pandas.DataFrame
        Weighted correlation matrix (p x p) with variable names.
    """
    # Convert inputs
    X = df.to_numpy()
    w = np.asarray(weights, dtype=float)
    
    # Normalize weights
    w_norm = w / w.sum()
    
    # Weighted mean
    mu = np.average(X, axis=0, weights=w_norm)
    
    # Centered and weighted data
    Xc = X - mu
    Xw = Xc * np.sqrt(w_norm)[:, np.newaxis]
    
    # Weighted covariance
    denom = 1.0 - np.sum(w_norm ** 2)
    cov_matrix = (Xw.T @ Xw) / denom

    # Standard deviations
    std_dev = np.sqrt(np.diag(cov_matrix))

    # Outer product of std_dev for normalization
    outer_std = np.outer(std_dev, std_dev)
    
    # Correlation matrix
    cor_matrix = cov_matrix / outer_std
    
    # Ensure diagonals are exactly 1.0 (numerical stability)
    np.fill_diagonal(cor_matrix, 1.0)
    
    # Return as labeled DataFrame
    cor_df = pd.DataFrame(cor_matrix, index=df.columns, columns=df.columns)
    
    return cor_df

#Define the moving average function
@njit
def ewma_vol(x, lam, start):
    """
    Compute EWMA volatility estimate assuming mean = 0.

    Parameters:
    - x: np.ndarray, 1D array of residuals
    - lam: float, smoothing factor (e.g., 0.94)
    - start: int, number of initial observations for variance estimate

    Returns:
    - vol: np.ndarray, same size as x, with np.nan for first `start` elements
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    vol = np.full(n, np.nan)

    if n <= start:
        return vol

    # Compute initial variance from the first `start` observations (skip NaNs)
    valid_x = x[:start]
    mask = ~np.isnan(valid_x)
    count = np.sum(mask)
    if count <= 1:
        return vol

    initial_var = np.sum(valid_x[mask] ** 2) / (count - 1)
    var = np.empty(n)
    var[:start] = np.nan
    var[start] = initial_var

    # Iteratively Updating EWMA Vol
    for i in range(start + 1, n):
        if np.isnan(x[i - 1]):
            var[i] = var[i - 1]
        else:
            #MA over squared residuals
            var[i] = lam * var[i - 1] + (1 - lam) * x[i - 1] ** 2

    vol = np.sqrt(var)
    return vol

#wrapper that pandas.apply() can call
def ewma_wrapper(series):
    """
    series → numpy → Numba → Series
    """
    arr = series.values.astype(float)        # ensure float array
    out = ewma_vol(arr, 
                   0.5**(1/hl_stock_var), initial_var_obs)          
    return pd.Series(out, index=series.index)


#%% Get Preliminaries
#List of Stock Features
features = GF.get_features(exclude_poor_coverage = True)

ff12_features = ['ff12_BusEq', 'ff12_Chems', 'ff12_Durbl', 
                 'ff12_Enrgy', 'ff12_Hlth', 'ff12_Manuf', 
                 'ff12_Money', 'ff12_NoDur', 'ff12_Other', 
                 'ff12_Shops', 'ff12_Telcm', 'ff12_Utils']

#%% Load DataSets

#Connect to DataBases
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_clean.db")
crsp_daily = sqlite3.connect(database = path + "Data/crsp_daily.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")

#Select Time Frame (Chunk) to read in
start_date = "1990-01-01"
end_date = "2024-12-31"

sp500_ids = list(pd.read_sql_query("SELECT * FROM SP500_Constituents_alltime",
                              con = SP500_Constituents)['id']
                 )

sp500_ids = ', '.join(str(x) for x in sp500_ids)
SP500_Constituents.close()

# =============================================================================
#                           Read in Data
# =============================================================================
print(f"Reading in Factors. Range: {start_date} to {end_date}")

#Build query vector for features:
#   Set Empty String
query_features =""
#   Fill the Empty String with the features
for feature in features + ff12_features:
    query_features = query_features + feature + ", "
#   Delete last comma and space to avoid syntax errors
query_features = query_features[:-2]
#   Build final query vector
query = ("SELECT id, eom, sic, ff49, size_grp, me, ret_exc, crsp_exchcd, in_sp500, " 
         + query_features 
         +" FROM Factors_processed " 
         +f"WHERE eom BETWEEN '{start_date}' AND '{end_date}' "
         +f"AND id IN ({sp500_ids})" #Filter for CRSP observations (id <= 99999)
         # Not required as I only have CRSP Data in my DataSet anyway
         )

# Read in JKP characteristics data.
chars  = pd.read_sql_query(query, 
                           con=JKP_Factors,
                           parse_dates=['eom']
                           )

print("    Complete.")

#%%

# =============================================================================
#                           Feature Ranks
# =============================================================================
"""
Performs feature standardization by converting features to percentile ranks within each time period.

For each feature:
1. Converts the feature values to float type for precision
2. Calculates empirical CDF (percentile ranks) within each 'eom' group, 
   so values are ranked relative to others in the same time period
3. Preserves exact zeros by setting them back to 0 after ranking
   (since ECDF would otherwise give them small positive values)

This normalization makes features comparable across different scales and distributions.
Operates in-place on the DataFrame.
"""    
ranked = chars.groupby("eom")[features].rank(pct=True) #rank is faster than using custom ecd_transform() and gives the same result

# Restore zeros
mask = chars[features] == 0.0
ranked[mask] = 0

#Overwrite feature data
chars[features] = ranked

#Fill missing values with 0
chars[features] = chars[features].apply(lambda x: x.fillna(0.5))

#Save Workspace
del ranked, mask

print("Feature Rank Complete.")

# =============================================================================
#                               Daily Data
# =============================================================================
print(f"Reading in Daily Returns. Range: {start_date} to {end_date}")

id_list = ', '.join(str(x) for x in chars['id'].unique())
#Read in Data
#   Query
query_daily = ("SELECT permno as id, date, ret_excess as ret_exc FROM d_ret_ex "
+ f"WHERE date BETWEEN '{start_date}' AND '{end_date}'" 
#+  "AND id <= 99999" #(only CRSP Data in my Dataset anyway)
+f"AND id IN ({id_list})"
)
#   Read in Data
daily = pd.read_sql_query(query_daily,
                          con = crsp_daily,
                          parse_dates = {'date'}
                          )

# Filter
n_start = len(daily)
pct_ret_missing = daily["ret_exc"].isna().mean() * 100
print(f"   Non-missing ret_exc excludes {pct_ret_missing:.2f}% of the observations")
daily = daily[(~daily['ret_exc'].isna())]

# Create end-of-month column
daily['eom'] = daily['date'] + pd.offsets.MonthEnd(0)

print("    Complete.")

# =============================================================================
#                           Cluster Data
# =============================================================================
print("Reading in Cluster Labels Data")
cluster_labels = pd.read_csv(path + "Data/cluster_labels_processed.csv")
print("    Complete.")

#%% Cluster Ranks
"""
Compute each stock's rank of a cluster of characteristics in the cross-section
for all valid observations
"""
#Get Valid Observations
cluster_data_m = chars[['id', 'eom', 'size_grp'] + features + ff12_features]

#Rename Object
#cluster_data_m = chars
#del chars #Note: Modifying Chars & Renaming Saves Memory

#Get Cluster Label Names
clusters = cluster_labels['cluster'].unique().tolist()
print("Cluster Labels are the following\n", clusters)

# Compute each stock's cluster ranks
cluster_ranks = build_cluster_ranks(cluster_data_m, cluster_labels, clusters, features)

#Save Memory
del cluster_labels
"""
Overwrite DataFrame and throw out Characteristics Data for Cluster Ranks
"""
# Add ranks to cluster_data_m
cluster_data_m = cluster_data_m[['id', 'eom', 'size_grp'] + ff12_features]

# Lead eom
cluster_data_m['eom_lead'] = (pd.to_datetime(cluster_data_m['eom']) + pd.offsets.MonthEnd(1))

# Combine with cluster_ranks_df
cluster_data_m = pd.concat([cluster_data_m, cluster_ranks], axis=1)

print("Cluster Ranks Completed.")


#Standardize Cluster Ranks in the cross-section
cluster_data_m[clusters] = (cluster_data_m
                            .groupby('eom')[clusters]
                            .transform(lambda x: (x - x.mean()) / x.std())
                            )

print("Cluster Cross-Sectional Z-Scores Completed.")

#%% Prepare DataFrame for Daily Stock Return Regression
# =============================================================================
#                           Daily Return Regression
# =============================================================================
"""
Merge 1 month lagged Cluster Ranks to daily Stock Return Data for the 
daily return regression.
"""
#Prepare daily return data
daily = daily[daily['date'] >= cluster_data_m['eom'].min()]

#Merge daily return data with cluster data
#   ranks from previous month are used in the regression, i.e. data was observable
cluster_data_d = (pd.merge(daily,cluster_data_m, left_on = ['id','eom'], right_on = ['id','eom_lead'], 
                          how='inner', suffixes = ("","_merge"))
                  .pipe(lambda df: df.drop(df.filter(regex='_merge$').columns, axis=1))
                  .drop(columns = ["eom_lead"])
                  )

#Rename Object
#cluster_data_d = daily
#del daily #Note: Modifying Chars & Renaming Saves Memory

# Drop rows with any missing data
cluster_data_d.dropna(inplace=True)

print("Daily Stock Return DataFrame Complete.") 

#%% Daily Regression
"""
Conduct the daily stock return regression.
Store the OLS Residual and the OLS-Coefficients, i.e. the daily estimated factor returns
"""
# Prepare column names
factor_cols = ff12_features + clusters
X_cols = factor_cols
y_col = 'ret_exc'

# Extract unique dates
unique_dates = cluster_data_d['date'].unique()

# Prepare result containers
all_residuals = []
all_coefs = []

# Get numerical objects
X_data = cluster_data_d[factor_cols].values
y_data = cluster_data_d[y_col].values
dates = cluster_data_d['date'].values
ids = cluster_data_d['id'].values

#Save Memory
del cluster_data_d

#Compute the Daily Regression
for date in tqdm([date for date in unique_dates], desc="Calculating Daily Return Regression"):
    # Boolean mask for the date
    mask = (dates == date)
    
    #numerical objects
    X = X_data[mask]
    y = y_data[mask]
    id_subset = ids[mask]

    # OLS
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(XtX) @ Xty  # fallback to pseudo-inverse

    # Compute predictions and residuals
    y_pred = X @ coef
    residuals = y - y_pred

    # Save results
    all_residuals.append(pd.DataFrame({
        'id': id_subset,
        'date': date,
        'residual': residuals
    }))
    all_coefs.append(pd.Series(coef, name=date))

#Save Workspace
del X_data, y_data, dates, ids

# DataFrame with Daily OLS-Residuals
spec_risk = pd.concat(all_residuals, axis=0)
spec_risk = spec_risk.sort_values(by=['id', 'date']).reset_index(drop=True)

#Save Workspace
del all_residuals

# Daily Factor Returns DataFrame
fct_ret = (pd.DataFrame(all_coefs).reset_index()
           .set_axis(['date'] + factor_cols, axis=1)
           .sort_values(by = 'date', ignore_index=True)
           )
#Store Dates to be able to delete fct_ret later to save workspace
fct_dates = fct_ret['date']

#Save Workspace
del all_coefs

print("Factor Returns Completed.")

#%% Factor Covariances
"""
Compute the Daily Factor Covariances and the Daily Squared OLS Residuals
(Stock Specific Risk) using exponentially weighted moving averages for every
last trading day of the month.

For monthly conversion, this data must be scaled by 21.

To estimate
the idiosyncratic variance we require at least 200 non-missing observations within the last
252 trading days, and otherwise, we use the median idiosyncratic variance of stocks within
the same size group.
"""

#Get Dates at which the Barra VarCov is computed (calc_dates)
ref_date = fct_ret['date'].iloc[obs] #at least 2520 trading days of lookback data
min_date = ref_date.to_period('M').to_timestamp() - pd.offsets.MonthEnd(1)
calc_dates = sorted(cluster_data_m[cluster_data_m['eom'] >= min_date]['eom'].unique()) # monthly dates for the VarCov
calc_dates = [d for d in calc_dates if d >= settings['rolling_window']['trading_month'] - pd.offsets.MonthEnd(1)]

# =============================================================================
#                Factor Covariance Matrix V(f_{t+1})
# =============================================================================

# Initialize factor covariance estimation
factor_cov_est = {}

for d in tqdm([date for date in calc_dates], desc="Calculating factor covariances"):
    
    # Determine first observation date
    eligible_dates = fct_ret[fct_ret['date'] <= d]['date']
    first_obs = eligible_dates.iloc[-obs:].min()
    
    # Filter covariance data
    cov_data = fct_ret[(fct_ret['date'] >= first_obs) & (fct_ret['date'] <= d)]
    
    t = len(cov_data)
    #====================
    # JKMP IMPLEMENTATION
    #=====================
    #Weighted Correlation Matrix
    cor_est = weighted_cor_wt(cov_data.drop('date',axis=1),w_cov[-t:])
    #Weighted Variance Matrix
    var_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_var[-t:])    
    
    # Calculate diagonal matrix with standard deviations on diagonal
    sd_diag = np.diag(np.sqrt(np.diag(var_est)))

    # Get Covariance Matrix by multiplying correlation matrix with standard deviation
    cov_est = (sd_diag @ cor_est @ sd_diag).set_index(cov_data.drop('date', axis=1).columns)
    cov_est.columns = cov_data.drop('date', axis=1).columns  
    
    #===================
    # My Implementation
    #===================
    """    
    The Problem with my implementation is that it can give negative variances
    for a stock when computing the barra covariance matrix S @ cov_est @ S' + diag(resid**2)
    
    cov_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_cov[-t:])
    var_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_var[-t:])
    cov_est.values[[range(len(cov_est))], [range(len(cov_est))]] = var_est.values.diagonal()
    
    
    """
    factor_cov_est[d] = cov_est
    
#Save Workspace
del fct_ret

# =============================================================================
#               Idiosyncratic Variance diag(epsilon_{t+1})
# =============================================================================

# Compute exponentially weighted moving average to get MONTHLY squared OLS residual
spec_risk['res_vol'] = (
    spec_risk
      .groupby('id')['residual']
      .apply(ewma_wrapper)
      .values
)

#--- Require that the observation has at least 200 non-missing oversations out of the last 252 trading days
# Create td_range dataframe with date and td_252d columns (252-day lag)
td_range = pd.DataFrame({'date': fct_dates,
                         'td_252d': fct_dates.shift(252)
                         })

# Merge spec_risk with td_range on date column
spec_risk = pd.merge(spec_risk, td_range, on='date', how='left')

# Add date_200d column with 200-day lag grouped by id
spec_risk['date_200d'] = spec_risk.groupby('id')['date'].shift(200)

# Filter rows where date_200d >= td_252d and res_vol is not NA
spec_risk = spec_risk[
    (spec_risk['date_200d'] >= spec_risk['td_252d']) & 
    (spec_risk['res_vol'].notna())
][['id', 'date', 'res_vol']]


#--- Extract specific risk by month end (note, these are still daily volatilities, we
#       only keep the end of month observation. Upscaling to monthly volatility happens
#       in the next section )
# Create eom_ret column (end of month date)
spec_risk['eom_ret'] = spec_risk['date'] + pd.offsets.MonthEnd(0)

# Create max_date column (maximum date for each id and eom_ret combination)
spec_risk['max_date'] = spec_risk.groupby(['id', 'eom_ret'])['date'].transform('max')

# Filter to keep only rows where date equals max_date, and select specific columns
spec_risk_m = spec_risk[spec_risk['date'] == spec_risk['max_date']][['id', 'eom_ret', 'res_vol']].copy()

# Rename eom_ret to eom
spec_risk_m = spec_risk_m.rename(columns={'eom_ret': 'eom'})

#Save Workspace
del spec_risk

#%% Barra Covariance Matrix
# =============================================================================
#               	Barra Covariance Matrix
# =============================================================================
"""
Compute the Barra Covariance Matrix as in (37). Note: Only the relevant objects
to do the final computation in (37) are saved.
"""
#Initialise Object
barra_cov = {}

#Loop over Dates
for d in tqdm(calc_dates, desc="Calculating Barra Cov"):  
    # Filter cluster_data_m for current date
    char_data = cluster_data_m[cluster_data_m['eom'] == d].copy()
    
    # Merge with specific risk data
    char_data = char_data.merge(spec_risk_m, on=['id', 'eom'], how='left')
    
    # Calculate median res_vol by size group and date
    char_data['med_res_vol'] = char_data.groupby(['size_grp', 'eom'])['res_vol'].transform(
        lambda x: x.median(skipna=True)
    )
    
    # Handle edge case where med_res_vol is still NA
    if char_data['med_res_vol'].isna().any():
        char_data['med_res_vol_all'] = char_data.groupby('eom')['res_vol'].transform(
            lambda x: x.median(skipna=True)
        )
        char_data.loc[char_data['med_res_vol'].isna(), 'med_res_vol'] = char_data['med_res_vol_all']
    
    # Impute missing res_vol with median values
    char_data.loc[char_data['res_vol'].isna(), 'res_vol'] = char_data['med_res_vol']
    
    # Get factor covariance matrix and monthly scaling (21 trading days in a month))
    fct_cov = factor_cov_est[d] * 21
    
    # Sort by id to ensure alignment
    char_data = char_data.sort_values('id')
    
    # Create factor loading matrix
    X = char_data[fct_cov.columns].values
    X = pd.DataFrame(X, index=char_data['id'].astype(int), columns=fct_cov.columns)
    
    # Create idiosyncratic variance vector (monthly scaling - 21 trading days in a month)
    ivol_vec = char_data['res_vol']**2 * 21
    ivol_vec.index = char_data['id'].astype(int)
        
    # Store results
    barra_cov[d] = {
        "fct_load": X,
        "fct_cov": fct_cov,
        "ivol_vec": ivol_vec
    }
#%% Save Dictionaries       
with open(path + '/Data/Barra_Cov.pkl', 'wb') as f:
    pickle.dump(barra_cov, f)
    
#%% Close remaining connections
JKP_Factors.close()
crsp_daily.close()