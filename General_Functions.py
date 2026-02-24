# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:04:35 2025

@author: Patrick
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
from scipy import stats

#%% Hyperparameter

def get_settings():

    settings = {'rolling_window':{'validation_periods':12, #12 months
                                  'window_size':10*12,  #10 years
                                  'test_size': 12, #1 year
                                  'trading_start': pd.to_datetime('2003-01-31'), #Dates onwards must be fully out of sample
                                  'tuning_fold': 1, # = 1 implies only the last window_size + validation_periods are used for cross-validation of the hyperparameters
                                  'trading_end': pd.to_datetime('2024-12-31') #Last trading period
                                  },
                   'RFF': {'p_vec':np.array([2**i for i in range(6,11)]),
                           'sigma_vec': np.array([0.5,1,10,50]),
                           'penalty_vec':np.array([1e-3, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
                           },
                   'gamma': 5.0
                   }
    
    return settings

#%%

#Get the List of Features
# Features ---------------------
def get_features(exclude_poor_coverage = True):
    """
    Original Features of the Jensen, Kelly & Pedersen (2023) Global factor dataset
    """
    features = [
      "age",                 "aliq_at",             "aliq_mat",            "ami_126d",           
      "at_be",               "at_gr1",              "at_me",               "at_turnover",        
      "be_gr1a",             "be_me",               "beta_60m",            "beta_dimson_21d",    
      "betabab_1260d",       "betadown_252d",       "bev_mev",             "bidaskhl_21d",       
      "capex_abn",           "capx_gr1",            "capx_gr2",            "capx_gr3",           
      "cash_at",             "chcsho_12m",          "coa_gr1a",            "col_gr1a",           
      "cop_at",              "cop_atl1",            "corr_1260d",          "coskew_21d",         
      "cowc_gr1a",           "dbnetis_at",          "debt_gr3",            "debt_me",            
      "dgp_dsale",           "div12m_me",           "dolvol_126d",         "dolvol_var_126d",    
      "dsale_dinv",          "dsale_drec",          "dsale_dsga",          "earnings_variability",
      "ebit_bev",            "ebit_sale",           "ebitda_mev",          "emp_gr1",            
      "eq_dur",              "eqnetis_at",          "eqnpo_12m",           "eqnpo_me",           
      "eqpo_me",             "f_score",             "fcf_me",              "fnl_gr1a",           
      "gp_at",               "gp_atl1",             "ival_me",             "inv_gr1",            
      "inv_gr1a",            "iskew_capm_21d",      "iskew_ff3_21d",       "iskew_hxz4_21d",     
      "ivol_capm_21d",       "ivol_capm_252d",      "ivol_ff3_21d",        "ivol_hxz4_21d",      
      "kz_index",            "lnoa_gr1a",           "lti_gr1a",            "market_equity",      
      "mispricing_mgmt",     "mispricing_perf",     "ncoa_gr1a",           "ncol_gr1a",          
      "netdebt_me",          "netis_at",            "nfna_gr1a",           "ni_ar1",             
      "ni_be",               "ni_inc8q",            "ni_ivol",             "ni_me",              
      "niq_at",              "niq_at_chg1",         "niq_be",              "niq_be_chg1",        
      "niq_su",              "nncoa_gr1a",          "noa_at",              "noa_gr1a",           
      "o_score",             "oaccruals_at",        "oaccruals_ni",        "ocf_at",             
      "ocf_at_chg1",         "ocf_me",              "ocfq_saleq_std",      "op_at",              
      "op_atl1",             "ope_be",              "ope_bel1",            "opex_at",            
      "pi_nix",              "ppeinv_gr1a",         "prc",                 "prc_highprc_252d",   
      "qmj",                 "qmj_growth",          "qmj_prof",            "qmj_safety",         
      "rd_me",               "rd_sale",             "rd5_at",              "resff3_12_1",        
      "resff3_6_1",          "ret_1_0",             "ret_12_1",            "ret_12_7",           
      "ret_3_1",             "ret_6_1",             "ret_60_12",           "ret_9_1",            
      "rmax1_21d",           "rmax5_21d",           "rmax5_rvol_21d",      "rskew_21d",          
      "rvol_21d",            "sale_bev",            "sale_emp_gr1",        "sale_gr1",           
      "sale_gr3",            "sale_me",             "saleq_gr1",           "saleq_su",           
      "seas_1_1an",          "seas_1_1na",          "seas_11_15an",        "seas_11_15na",       
      "seas_16_20an",        "seas_16_20na",        "seas_2_5an",          "seas_2_5na",         
      "seas_6_10an",         "seas_6_10na",         "sti_gr1a",            "taccruals_at",       
      "taccruals_ni",        "tangibility",         "tax_gr1a",            "turnover_126d",      
      "turnover_var_126d",   "z_score",             "zero_trades_126d",    "zero_trades_21d",    
      "zero_trades_252d",
      "rvol_252d"
    ]
    
    # Exclude features without sufficient coverage
    feat_excl = ["capex_abn", "capx_gr2", "capx_gr3", "debt_gr3", "dgp_dsale",           
                   "dsale_dinv", "dsale_drec", "dsale_dsga", "earnings_variability", "eqnetis_at",          
                   "eqnpo_me", "eqpo_me", "f_score", "iskew_hxz4_21d", "ivol_hxz4_21d",       
                   "netis_at", "ni_ar1", "ni_inc8q", "ni_ivol", "niq_at", "niq_at_chg1", "niq_be", 
                   "niq_be_chg1", "niq_su", "ocfq_saleq_std", "qmj", "qmj_growth", "rd_me", 
                   "rd_sale", "rd5_at", "resff3_12_1", "resff3_6_1", "sale_gr3", "saleq_gr1", 
                   "saleq_su", "seas_16_20an", "seas_16_20na", "sti_gr1a", "z_score"
    ]
    if exclude_poor_coverage:
        # Filter out the excluded features
        features = [feature for feature in features if feature not in feat_excl]
    
    return features


def get_signals(exclude_poor_coverage = True):
    industry_codes = ['ff12_BusEq',
                      'ff12_Chems',
                      'ff12_Durbl',
                      'ff12_Enrgy',
                      'ff12_Hlth',
                      'ff12_Manuf',
                      'ff12_Money',
                      'ff12_NoDur',
                      'ff12_Other',
                      'ff12_Shops',
                      'ff12_Telcm',
                      'ff12_Utils'
                      ]
    signals = [get_features(exclude_poor_coverage),industry_codes]
    
    return signals
    

#%%
def long_horizon_ret(data, h=1, impute="zero"):
    """
    Compute long-horizon returns.
    
    Parameters:
      data: pandas DataFrame with at least columns 'id', 'eom', 'ret_exc'
      h: int, horizon (number of future periods)
      impute: str, imputation method in {"zero", "mean", "median"}
      
    Returns:
      DataFrame with future return columns ret_ld1, ..., ret_ld{h}.
    """
    # Ensure that the input data is not modified
    data = data.copy()
    
    #Ensure 'eom' is datetime
    data['eom'] = pd.to_datetime(data['eom'])
    
    # Only consider rows where ret_exc is not NA.
    valid_data = data.dropna(subset=["ret_exc"])
    
    # Get unique dates (for merging) present in valid_data.
    dates = valid_data[['eom']].drop_duplicates()
    dates = dates.rename(columns={'eom': 'merge_date'})
    
    # For each id, find the start and end dates where ret_exc is not missing.
    ids = valid_data.groupby('id')['eom'].agg(start='min', end='max').reset_index()
    
    # Create a cross between each security’s valid date range and the unique dates.
    # For each id, select the dates between start and end.
    id_dates = ids.merge(dates, how="cross")
    id_dates = id_dates[(id_dates['merge_date'] >= id_dates['start']) & 
                        (id_dates['merge_date'] <= id_dates['end'])]
    
    # Remove the extra start/end columns.
    id_dates = id_dates.drop(columns=["start", "end"]).rename(columns={"merge_date": "eom"})
    
    # Merge the full panel with the original return data.
    full_ret = pd.merge(id_dates, data[['id', 'eom', 'ret_exc']], on=["id", "eom"], how="left")
    full_ret = full_ret.sort_values(by=["id", "eom"]).reset_index(drop=True)
    
    # Create the lead return columns ret_ld1, ..., ret_ldh within each id group.
    for l in range(1, h+1):
        full_ret[f"ret_ld{l}"] = full_ret.groupby("id")["ret_exc"].shift(-l)
    
    # Drop the original ret_exc column.
    full_ret = full_ret.drop(columns=["ret_exc"])
    
    # Identify rows where all lead returns are missing.
    lead_cols = [f"ret_ld{l}" for l in range(1, h+1)]
    all_missing = full_ret[lead_cols].isna().sum(axis=1) == h
    perc_all_missing = all_missing.mean() * 100
    print(f"All missing excludes {perc_all_missing:.2f}% of the observations")
    
    # Keep only rows where not all lead returns are missing.
    full_ret = full_ret[~all_missing].reset_index(drop=True)
    
    # Impute missing values in the lead columns based on the specified method.
    if impute == "zero":
        full_ret[lead_cols] = full_ret[lead_cols].fillna(0)
    elif impute == "mean":
        # Replace NA with the mean of the column within each month (eom group).
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.mean()))
    elif impute == "median":
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.median()))
    
    return full_ret


#%%

# Wealth Calculation Function
def wealth_func(wealth_end, end, market, risk_free):
    """
    Compute the wealth trajectory backwards from a given initial wealth
    level. Since backtesting works backwards, for a given initial wealth level
    we need to know how our strategy would have performed in the past to evaluate
    it.
    
    Wealth is assumed to grow exogenously at the market return.
    
    Parameters:
      wealth_end: final wealth value
      end: final date (as a pd.Timestamp)
      market: DataFrame with market returns, containing 'eom_ret' and 'mkt_vw_exc'
      risk_free: DataFrame with risk-free data, containing 'eom' and 'rf'
      
    Returns:
      A DataFrame with columns: 'eom' (end-of-month), 'wealth', and 'mu_ld1' (total return for the period)
    """
    # Rename and merge
    risk_free = risk_free.rename(columns={"eom": "eom_ret"})
    wealth = pd.merge(risk_free[["eom_ret", "rf"]], market, on="eom_ret", how="left")

    # Compute total return
    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    # Filter for dates up to 'end'
    wealth = wealth[wealth["eom_ret"] <= end]

    # Sort descending by date and compute cumulative wealth backward
    wealth = wealth.sort_values("eom_ret", ascending=False)
    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end

    # Final output formatting
    wealth["eom"] = (wealth["eom_ret"].dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1))
    wealth["mu"] = wealth["tret"]

    result = wealth[["eom", "wealth", "mu"]].copy()

    # Add one row for the exact 'end' date
    row = pd.DataFrame([{"eom": end, "wealth": wealth_end, "mu": np.nan}])
    result = pd.concat([result, row], ignore_index=True)

    # Sort chronologically
    result = result.sort_values("eom").reset_index(drop=True)

    return result

def create_cov(x, ids=None):
    """
    Create the Barra Covariance Matrix 
    
    Parameters:
    x (dict): Dictionary containing 'fct_load', 'ivol_vec', and 'fct_cov'
    ids (array-like, optional): List of Stock IDs to subset the data
    
    Returns:
    numpy.ndarray: The computed covariance matrix
    """
    ################## Compute the Covariance Matrix ##########################
    # Extract the relevant loadings and ivol
    if ids is None:
        load = x['fct_load']
        ivol = x['ivol_vec']
    else:
        # Convert ids to strings to match R's behavior with as.character()
        load = x['fct_load'].loc[[i for i in ids]]
        ivol = x['ivol_vec'].loc[[i for i in ids]]
    
    # Create the covariance matrix
    sigma = load @ x['fct_cov'] @ load.T + np.diag(ivol) 
    
    ################## Error Correction ##########################
    """
    In case a stock variance is negative. Can happen as (37) is not exactly an
    equality. Due to the EWMA the residuals are not uncorrelated with the fitted values.
    """
    if min(np.diag(sigma)) < 0:
        # Get diagonal indices
        diag_indices = np.arange(len(sigma))
        print("Warning: Negative Variances:", diag_indices)
        
        """
        # Extract the diagonal
        diag_values = sigma.values[diag_indices, diag_indices]
        
        # Find the minimum of the positive diagonal values and use them to replac ethe negatives
        positive_diag = diag_values[diag_values > 0]
        if len(positive_diag) > 0:
            min_positive = positive_diag.min()
        
            # Create a boolean mask where diagonal elements are zero
            mask = diag_values < 0
        
            # Replace zero diagonal elements with min_positive
            sigma.values[diag_indices[mask], diag_indices[mask]] = min_positive
        """
            
    return sigma

#%%

#Categories Industries According to Fama-French 12 Industries
#Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
#Categories can also be found in the data folder
def categorize_sic(sic):
    """
    Assigns Fama-French 12 industry classification based on SIC codes.

    Parameters:
    -----------
    chars : pd.DataFrame
        DataFrame containing a 'sic' column with Standard Industrial Classification codes
        
    Returns:
    --------
    pd.Series
        Series containing industry classifications according to Fama-French 12 categories:
        - NoDur: Non-durable Consumer Goods
        - Durbl: Durable Consumer Goods
        - Manuf: Manufacturing
        - Enrgy: Energy
        - Chems: Chemicals
        - BusEq: Business Equipment
        - Telcm: Telecommunications
        - Utils: Utilities
        - Shops: Shops (Wholesale/Retail)
        - Hlth: Healthcare
        - Money: Financial Services
        - Other: All other industries
        
    Note:
    -----
    Follows the standard Fama-French 12 industry classification scheme based on SIC code ranges.
    """
    # Non-Durables
    if ((100 <= sic <= 999) or
        (2000 <= sic <= 2399) or
        (2700 <= sic <= 2749) or
        (2770 <= sic <= 2799) or
        (3100 <= sic <= 3199) or
        (3940 <= sic <= 3989)):
        return "NoDur"
    
    # Durables
    elif ((2500 <= sic <= 2519) or
          (3630 <= sic <= 3659) or
          (sic in [3710, 3711, 3714, 3716, 3750, 3751, 3792]) or
          (3900 <= sic <= 3939) or
          (3990 <= sic <= 3999)):
        return "Durbl"
    
    # Manufacturing
    elif ((2520 <= sic <= 2589) or
          (2600 <= sic <= 2699) or
          (2750 <= sic <= 2769) or
          (3000 <= sic <= 3099) or
          (3200 <= sic <= 3569) or
          (3580 <= sic <= 3629) or
          (3700 <= sic <= 3709) or
          (3712 <= sic <= 3713) or
          (sic in [3715]) or
          (3717 <= sic <= 3749) or
          (3752 <= sic <= 3791) or
          (3793 <= sic <= 3799) or
          (3830 <= sic <= 3839) or
          (3860 <= sic <= 3899)):
        return "Manuf"
    
    # Energy
    elif ((1200 <= sic <= 1399) or
          (2900 <= sic <= 2999)):
        return "Enrgy"
    
    # Chemicals
    elif ((2800 <= sic <= 2829) or
          (2840 <= sic <= 2899)):
        return "Chems"
    
    # Business Equipment
    elif ((3570 <= sic <= 3579) or
          (3660 <= sic <= 3692) or
          (3694 <= sic <= 3699) or
          (3810 <= sic <= 3829) or
          (7370 <= sic <= 7379)):
        return "BusEq"
    
    # Telecommunications
    elif (4800 <= sic <= 4899):
        return "Telcm"
    
    # Utilities
    elif (4900 <= sic <= 4949):
        return "Utils"
    
    # Shops
    elif ((5000 <= sic <= 5999) or
          (7200 <= sic <= 7299) or
          (7600 <= sic <= 7699)):
        return "Shops"
    
    # Health
    elif ((2830 <= sic <= 2839) or
          (sic == 3693) or
          (3840 <= sic <= 3859) or
          (8000 <= sic <= 8099)):
        return "Hlth"
    
    # Money
    elif (6000 <= sic <= 6999):
        return "Money"
    
    # Other
    else:
        return "Other"
    
#%%
def portfolio_return_BFGS(logits, pi_tm1, gt, Sigma, KL, wealth, return_predictions, gamma):
    
    #Get current portfolio (in levels)
    pi_t = np.exp(logits)
    pi_t /= np.sum(pi_t)
    
    #Compute Revenue
    revenue = pi_t.T @ return_predictions 
    
    #Compute Variance penalty
    var_pen = gamma/2 * pi_t.T @ Sigma @ pi_t
    
    #Compute transaction costs
    change_pf = pi_t-pi_tm1 
    tc = 0.5* wealth * np.sum(KL * change_pf**2)
    
    return -(revenue - tc - var_pen)

#%% Return Prediction Data

def load_retpred_data(
    db_signals: sqlite3.Connection,
    db_sp500: sqlite3.Connection,
    settings: dict,
    target: str,
    rank_signals: bool,
    trade_start: str,
    trade_end: str,
    continuous_signals: list, # list of signals that are NOT one-hot encodings (require cross-sectional standardisation)
    fill_missing_values: bool,
    CRSP: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, int, list[str], int, int]:
    """
    Reads, merges, and preprocesses signal and target data for a rolling window analysis.

    Notes
    -----
    The database tables ('Signals_Rank' / 'Signals_ZScore') are assumed to be computed on the
    full CRSP universe. If `CRSP=False` (i.e., you subset to S&P 500 constituents), the
    cross-sectional standardisation must be recomputed within the restricted universe at each
    month-end ('eom'). This function therefore re-standardises the signal columns by 'eom'
    after applying the S&P 500 filter:
      - If `rank_signals=True`: re-ranks each signal within 'eom' to [-1, 1].
      - If `rank_signals=False`: re-zscores each signal within 'eom'.

    Args:
        db_signals: An active SQLite database connection object.
        settings: A dictionary containing 'rolling_window' configuration.
        target: Prediction target column name (from Factors_processed).
        rank_signals: If True, queries the 'Signals_Rank' table; otherwise queries the 'Signals_ZScore' table.
        trade_start: First trading date (YYYY-MM-DD). Used to construct monthly trading dates (month-end).
        trade_end: Last trading date (YYYY-MM-DD). Used to construct monthly trading dates (month-end).
        fill_missing_values: If True, fills NaNs of feature columns with zero (industry classifiers have no missing values).
        CRSP: If True, returns the full CRSP universe (function behaves as before).
              If False, restricts the sample to S&P 500 constituents ('SP500_Constituents_FL')
              and recomputes cross-sectional standardisation of signals within that subset.

    Returns:
        A tuple containing:
        - df_signals: The merged and cleaned DataFrame.
        - signal_months: A numpy array of unique 'eom' dates in the signals data.
        - trade_idx: The index in signal_months where the rolling window process starts.
        - feat_cols: List of feature column names used as signals.
        - window_size: Rolling window size (periods).
        - validation_size: Validation window size (periods).
    """

    # --- Helper Functions for cross-sectional standardisation ---
    def rank_standardise(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return group[cols].apply(lambda s: s.rank(method="average", pct=True) * 2 - 1)

    def Zscore_standardise(
        g: pd.DataFrame,
        cols,
        ddof: int = 0,
        clip: float | None = None,
        min_n: int = 5,
    ):
        if isinstance(cols, str):
            cols = [cols]

        x = g[cols].astype(float)

        n = x.notna().sum(axis=0)
        mu = x.mean(axis=0, skipna=True)
        sd = x.std(axis=0, ddof=ddof, skipna=True)

        ok = (n >= min_n) & (sd > 0) & np.isfinite(sd)

        z = pd.DataFrame(index=g.index, columns=cols, dtype=float)
        for c in cols:
            if ok.get(c, False):
                z[c] = (x[c] - mu[c]) / sd[c]
            else:
                z[c] = np.nan

        if clip is not None:
            z = z.clip(lower=-clip, upper=clip)

        return z[cols[0]] if len(cols) == 1 else z

    # 1. Configuration Extraction
    window_size     = settings["rolling_window"]["window_size"]
    validation_size = settings["rolling_window"]["validation_periods"]

    # Determine which signals table to query
    signal_table = "Signals_Rank" if rank_signals else "Signals_ZScore"

    # Trading Dates
    trading_dates = pd.date_range(trade_start, trade_end, freq="ME")

    # Calculate the required data before 1st trading date
    required_lag = window_size + validation_size + 1
    query_start_date = trading_dates[0] - pd.offsets.MonthEnd(required_lag)
    query_start_date_str = query_start_date.strftime("%Y-%m-%d")

    # Feature columns
    signals = get_signals()
    feat_cols = signals[0] + signals[1]

    # End date (include next month-end like original code)
    query_end_date_str = (pd.to_datetime(trade_end) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    # 2. Read Signals Data (always full universe as stored)
    query_signals = (
        f"SELECT * FROM {signal_table} "
        f"WHERE eom >= '{query_start_date_str}' "
        f"AND eom <= '{query_end_date_str}'"
    )
    df_signals = pd.read_sql_query(query_signals, con=db_signals, parse_dates={"eom"})

    # 3. Read Targets Data
    query_targets = (
        f"SELECT id, eom, {target} FROM Factors_processed "
        f"WHERE eom >= '{query_start_date_str}' "
        f"AND eom <= '{query_end_date_str}'"
    )
    df_targets = pd.read_sql_query(query_targets, con=db_signals, parse_dates={"eom"})

    # 4. Merge and Pre-process
    df_signals = (
        df_signals.merge(df_targets, on=["id", "eom"], how="left")
        .dropna(subset=[target])
        .sort_values(by=["eom", "id"])
    )

    if df_signals[target].isna().sum() > 0:
        print("ERROR: Missing Values in Target after merge and dropna. This is unexpected.")

    # 4b. Restrict to S&P 500 universe (when CRSP == False)
    if not CRSP:
        df_include = pd.read_sql_query(
            "SELECT * FROM SP500_Constituents_FL",
            con=db_sp500,
            parse_dates={"eom"},
        ).rename(columns={"PERMNO": "id"})

        df_signals = (
            df_signals.merge(df_include.assign(in_sp500=1), on=["id", "eom"], how="left")
            .loc[lambda x: x["in_sp500"] == 1]
            .drop(columns="in_sp500")
            .sort_values(by=["eom", "id"])
        )
        del df_include

        # --- Re-standardise within the restricted universe, by month ---
        # (because the stored tables were standardised on full CRSP)
        if rank_signals:
            # ranks to [-1, 1] within each eom
            df_signals[continuous_signals] = (
                df_signals.groupby("eom", group_keys=False)
                .apply(lambda g: rank_standardise(g, continuous_signals))
            )
        else:
            # z-scores within each eom
            df_signals[continuous_signals] = (
                df_signals.groupby("eom", group_keys=False)
                .apply(lambda g: Zscore_standardise(g, continuous_signals))
            )

    # 5. Fill Missing Values in Signals (one-hot encodings are never missing due to data_preprocessing.py)
    if fill_missing_values:
        df_signals[continuous_signals] = df_signals[continuous_signals].fillna(0)

    # 6. Calculate outputs
    signal_months = df_signals["eom"].sort_values().unique()
    trade_idx = np.searchsorted(signal_months, trading_dates[0])
    
    # 7. Sort Signals data by date and id
    df_signals = df_signals.sort_values(by = ['eom','id']).reset_index(drop = True)

    return df_signals, signal_months, trade_idx, feat_cols, window_size, validation_size

def load_signals_rollingwindow(
    db_conn: sqlite3.Connection,
    settings: dict,
    target: str,
    rank_signals: bool,
    trade_start: str,
    trade_end: str, 
    fill_missing_values: bool
) -> tuple[pd.DataFrame | None, np.ndarray | None, int | None]:
    """
    Reads, merges, and preprocesses signal and target data for a rolling window analysis.

    Args:
        db_conn: An active SQLite database connection object (e.g., to JKP_clean.db).
        settings: A dictionary containing 'rolling_window' configuration.
        target: Prediction Target
        rank_signals: If True, queries the 'Signals_Rank' table; otherwise,
                          queries the 'Signals_ZScore' table.
        fill_missing_values: If True, fills NaNs of features with zero (industry classifiers have no missing values).

    Returns:
        A tuple containing:
        - df_signals: The merged and cleaned DataFrame (or None on failure).
        - signal_months: A numpy array of unique 'eom' dates in the signals data (or None on failure).
        - trade_idx: The index in signal_months where the rolling window process starts (or None on failure).
    """
    # 1. Configuration Extraction
    window_size = settings['rolling_window']['window_size']
    validation_size = settings['rolling_window']['validation_periods']

    # Determine which signals table to query
    signal_table = "Signals_Rank" if rank_signals else "Signals_ZScore"

    # Trading Dates (Hardcoded end date from original snippet)
    trading_dates = pd.date_range(trade_start, trade_end, freq='ME')

    # Calculate the required data before 1st trading date
    required_lag = window_size + validation_size + 1
    query_start_date = trading_dates[0] - pd.offsets.MonthEnd(required_lag)
    query_start_date_str = query_start_date.strftime("%Y-%m-%d")
    
    #Feature columns
    signals = get_signals()
    feat_cols = signals[0] + signals[1]

    # 2. Read Signals Data
    query_signals = (f"SELECT * FROM {signal_table} "
                     f"WHERE eom >= '{query_start_date_str}' "
                     f"AND eom <= '{(pd.to_datetime(trade_end)+pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'"
                     )

    df_signals = pd.read_sql_query(
        query_signals,
        con=db_conn,
        parse_dates={'eom'}
    )

    # 3. Read Targets Data
    query_targets = (f"SELECT id, eom, {target} FROM Factors_processed "
                     f"WHERE eom >= '{query_start_date_str}' "
                     f"AND eom <= '{(pd.to_datetime(trade_end)+pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'"
                     )

    df_targets = pd.read_sql_query(
        query_targets,
        con=db_conn,
        parse_dates={'eom'}
    )

    # 4. Merge and Pre-process
    df_signals = (df_signals
                  .merge(df_targets, on=['id', 'eom'], how='left')
                  #!!!! BUG !!!! Will drop the last date of the sample.
                  # Drop rows where target (tr_m_sp500_ld1) is missing.
                  .dropna(subset=[target])
                  .sort_values(by=['eom', 'id'])
                 )

    # Sanity check from original code (should be 0 after dropna)
    if df_signals[target].isna().sum() > 0:
        print("ERROR: Missing Values in Target after merge and dropna. This is unexpected.")

    # 5. Fill Missing Values in Signals
    if fill_missing_values:
        df_signals[signals[0]] = df_signals[signals[0]].fillna(0)

    # 6. Calculate outputs
    signal_months = df_signals['eom'].sort_values().unique()

    # Find the index of the first trading date in the signal months array
    trade_idx = np.searchsorted(signal_months, trading_dates[0])

    return df_signals, signal_months, trade_idx, feat_cols, window_size, validation_size
