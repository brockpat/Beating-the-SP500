# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:19:23 2026

@author: patri
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

import pickle
import sqlite3

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from typing import Dict, List, Optional, Tuple, Any

import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Generals

# Trading Start & End
trading_start, trading_end = pd.to_datetime("2004-01-31"), pd.to_datetime("2024-12-31")

# DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_processed.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")
Models = sqlite3.connect(database = path + "Data/Predictions.db")
db_OtherPortfolios = sqlite3.connect(database = path + "Data/Other_Portfolios.db")

# Realised Returns & ME
df = pd.read_sql_query(("SELECT id, eom, me, tr_ld1, dolvol_126d FROM Factors_processed "
                        f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                        f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                        ),
                       con = JKP_Factors,
                       parse_dates = {'eom'}).sort_values(by = ['eom','id'])

df_returns = df[['id','eom','tr_ld1']]
df_me = df[['id','eom','me']]
df_dolvol = df[['id','eom','dolvol_126d']]
del df

# Names
df_names = pd.read_csv(path + "Data/CRSP_names.csv").drop_duplicates(subset = 'PERMNO')

# S&P500 Constituents
df_sp500_ids = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )

# SP500 ETF Benchmark
df_spy = pd.read_sql_query("SELECT * FROM SPY",
                           parse_dates = {'eom'},
                           con = Benchmarks)
df_spy = (df_spy[df_spy['eom'].between(trading_start, trading_end)]
              .assign(cumulative_return = lambda df: (1.0 + df['ret']).cumprod()
                      )
              ).reset_index(drop = True)

# risk-free rate
risk_free = (pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])
             .assign(rf = lambda df: df["RF"]/100)
             .assign(eom = lambda df: pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0))
             .get(['eom','rf'])
             )

# Kyle's Lambda (load one month before trading_start as lambda can be required at the BEGINNING of a month)
df_kl = (
    pd.read_sql_query(
        ("SELECT id, eom, lambda FROM Factors_processed "
         f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}'"),
        con=JKP_Factors,
        parse_dates={'eom'}
        )
    .sort_values(['id', 'eom'])
    .assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
    )

# Momentum Variables
df_mom = pd.read_sql_query(("SELECT id, eom, ret_1_0, ret_3_1, ret_6_1, ret_9_1, ret_12_1, ret_12_7, ret_60_12, in_sp500 FROM Factors_processed "
                           f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                           f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con = JKP_Factors,
                           parse_dates = {'eom'})
df_mom = df_mom.loc[df_mom['in_sp500'] == 1].drop('in_sp500', axis = 1)

# Wealth (AUM). load one month before trading_start as wealth can be required at the BEGINNING of a month)
df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
df_wealth = df_wealth.loc[(df_wealth['eom'] >= trading_start - pd.offsets.MonthEnd(1)) 
                          & 
                          (df_wealth['eom'] <= trading_end)
                          ].sort_values(by = 'eom')


#%% Functions: Performance Metrics

def meanRet_varRet(df, return_col):
    mu = df[return_col].mean()
    sigma = df[return_col].std(ddof=0)
    
    # Annualised
    return 12*mu, np.sqrt(12)*sigma

def SharpeRatio(df, risk_free, return_col):
    """
    Computes the annualised Sharpe Ratio of a time series of monthly returns.
    """
    df = (df
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col] - df['rf'])
          )

    #Sharpe Ratio
    mu_Sharpe, sigma_Sharpe = df['ret_exc'].mean(), df['ret_exc'].std() 
    Sharpe = np.sqrt(12) * (mu_Sharpe / sigma_Sharpe)
    
    return Sharpe

def InformationRatio(strategy, benchmark, return_col_strat, return_col_bench, risk_free):
    
    #Merge risk-free rate
    strategy = (strategy
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col_strat] - df['rf'])
          )
    
    benchmark = (benchmark
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col_bench] - df['rf'])
          )
    
    information_ratio = np.sqrt(12) * np.mean(strategy['ret_exc'] - benchmark['ret_exc'])/((strategy['ret_exc'] - benchmark['ret_exc']).std())
    
    return information_ratio

def MaxDrawdown(strategy, benchmark, return_col_strat, return_col_bench):
    """Calculates Maximum Drawdown from a return series."""
    # Convert returns to a cumulative wealth index
    comp_ret = (1 + strategy[return_col_strat]).cumprod()
    # Calculate the running maximum
    peaks = comp_ret.expanding(min_periods=1).max()
    # Calculate drawdown relative to the peak and return the minimum (most negative) value
    drawdown_strat = ((comp_ret / peaks) - 1).min()
    
    # Convert returns to a cumulative wealth index
    comp_ret = (1 + benchmark[return_col_bench]).cumprod()
    # Calculate the running maximum
    peaks = comp_ret.expanding(min_periods=1).max()
    # Calculate drawdown relative to the peak and return the minimum (most negative) value
    drawdown_bench = ((comp_ret / peaks) - 1).min()
    
    return drawdown_strat, drawdown_bench, (drawdown_strat - drawdown_bench)

def CaptureRatio(strategy, benchmark, return_col_strat, return_col_bench):
    """
    Calculates the Geometric Upside and Downside Capture Ratios.
    This is preferred over arithmetic mean for accuracy since going down 5% and
    back up 5% doesn't lead to being at the same level.
    
    If a fund has a downside capture ratio of 80%, then, during a period when 
    the market dropped 10%, the fund only lost 8%
    """
    
    # 1. Identify Downside Months (Benchmark < 0)
    down_mask = benchmark[return_col_bench] < 0
    strat_down = strategy[return_col_strat][down_mask]
    bench_down = benchmark[return_col_bench][down_mask]
    
    # 2. Identify Upside Months (Benchmark > 0)
    up_mask = benchmark[return_col_bench] > 0
    strat_up = strategy[return_col_strat][up_mask]
    bench_up = benchmark[return_col_bench][up_mask]

    # Helper function for Geometric Mean Return
    def geometric_mean(returns):
        # (Product of (1+r))^(1/n) - 1
        if len(returns) == 0: return np.nan
        compounded = np.prod(1 + returns)
        return compounded**(1 / len(returns)) - 1

    # 3. Calculate Geometric Means
    geo_avg_strat_down = geometric_mean(strat_down)
    geo_avg_bench_down = geometric_mean(bench_down)
    
    geo_avg_strat_up = geometric_mean(strat_up)
    geo_avg_bench_up = geometric_mean(bench_up)
    
    # 4. Calculate Means
    avg_strat_down = strat_down.mean()
    avg_bench_down = bench_down.mean()
    
    avg_strat_up = strat_up.mean()
    avg_bench_up = bench_up.mean()
    
    # 5. Calculate Ratios
    geo_downside_capture = geo_avg_strat_down / geo_avg_bench_down
    geo_upside_capture = geo_avg_strat_up / geo_avg_bench_up
    
    downside_capture = avg_strat_down / avg_bench_down
    upside_capture   = avg_strat_up / avg_bench_up
    
    return geo_downside_capture, geo_upside_capture, downside_capture, upside_capture

def capm_alpha(strategy, benchmark, return_col_strat, return_col_bench, risk_free):
    """Calculates Annualized Alpha and the Beta, setting alpha to 0 if not significant."""
    strategy = strategy.merge(risk_free, on = ['eom'], how = 'left').sort_values(by = 'eom')
    benchmark = benchmark.merge(risk_free, on = ['eom'], how = 'left').sort_values(by = 'eom')
    benchmark = benchmark.loc[benchmark['eom'].isin(strategy['eom'])]
                 
    y = strategy[return_col_strat] - strategy['rf']
    x = benchmark[return_col_bench]- benchmark['rf']
    
    # Linear Regression: y = alpha + beta * x
    # Note: linregress p_value is for the slope (beta). 
    # To get p_value for intercept (alpha), we use statsmodels or check significance manually.
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    alpha_monthly = model.params['const']
    beta = model.params.iloc[1] # The slope
    p_value_alpha = model.pvalues['const'] # The p-value for the intercept
    print(p_value_alpha)
    
    # Set alpha to zero if p-value is greater than 0.05
    #if p_value_alpha > 0.05:
    #    alpha_monthly = 0.0
    
    # Annualize Alpha
    alpha_annualized = alpha_monthly * 12 
    
    return alpha_annualized, p_value_alpha
                                        
def Turnover(strategy, aum):
    # 1. Compute per-asset absolute weight changes
    changes = (
        strategy
        .loc[:, ['eom', 'pi', 'pi_g_tm1']]
        .assign(abs_weight_change=lambda df: (df['pi'] - df['pi_g_tm1']).abs())
    )
    
    # 2. Aggregate to monthly gross turnover
    monthly_gross_turnover = (
        changes
        .groupby('eom', as_index=False)['abs_weight_change']
        .sum()
        .rename(columns={'abs_weight_change': 'gross_turnover'})
    )
    
    # 3. Compute AUM weights across time
    aum_weights = (
        aum
        .loc[:, ['eom', 'wealth']]
        .assign(aum_weight=lambda df: df['wealth'] / df['wealth'].sum())
    )
    
    # 4. AUM-weighted average turnover
    aum_weighted_turnover = (
        monthly_gross_turnover
        .merge(aum_weights, on='eom', how='left')
        .assign(weighted_turnover=lambda df: df['gross_turnover'] * df['aum_weight'])
        ['weighted_turnover']
        .sum()
    )      

    return monthly_gross_turnover['gross_turnover'].mean(), aum_weighted_turnover

#%% Functions: Other

def star_notation(p):
    """ 
    Used to indicate significance of results in tables
    """
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''

def long_short_portfolio(df, prediction_col,  # Predicted Returns
                         df_returns, # Realised Returns
                         df_me, # Market Equity for Value Weighting
                         long_cutoff = 0.9, short_cutoff = 0.1,
                         value_weighted = False,
                         long_only = False,
                         ):

    #===========================
    # Select Long & Short Stocks
    #===========================
    
    #Cross-Sectional Quantile cutoffs
    grouped = df.groupby('eom')[prediction_col]
    q_long = grouped.transform('quantile', long_cutoff)
    q_short= grouped.transform('quantile', short_cutoff)
    
    #---- Determine Positions ----
    #Conditions determining Long or Short
    conditions = [df[prediction_col] >=q_long,
                  df[prediction_col] <= q_short]
    
    #Numerical Position Value
    position = [1,-1]

    #Generating dataframe
    df_ls = df[['id','eom']]
    df_ls['position'] = np.select(conditions, position, default = 0)
    
    #Filter out zero positions
    df_ls = df_ls.loc[df_ls['position'] != 0]
    
    # If Long only
    if long_only:
        df_ls = df_ls.loc[df_ls['position']>0]
    
    #=====================================
    # Merge Market Equity & Future Return
    #=====================================
    
    # Merge market equity and future returns
    df_ls = (df_ls
             .merge(df_me, on = ['id','eom'], how = 'left')
             .merge(df_returns[['id','eom','tr_ld1']], on = ['id','eom'],
                                 how = 'left')
             )
    
    #===========================
    # Compute Portfolio weight
    #===========================
        
    if value_weighted:

        # Error Handling
        if df_ls['me'].isna().sum() > 0:
            print("ERROR: Missing market equity")
            
            return None, None
            
        #Compute weights
        df_ls = df_ls.assign(pi = lambda df:
                             df['me']/df.groupby(['eom','position'])['me'].transform('sum'))
    
    else: #equal weighted
        df_ls = df_ls.assign(pi = lambda df: 1/df.groupby(['eom','position'])['id'].transform('count'))

        
    # !!!! Very important: The information used is based on 'eom'.
    # The portfolio weight 'pi' is at the beginning of month of eom+1!!!!
    
    # Realign Date such that 'pi' is beginning of month at 'eom' (currently, in the data pi is at the end of month at 'eom')
    df_ls = df_ls.assign(eom = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
    
    # Rename 'tr_ld1' to 'tr' since it is the return at 'eom'
    df_ls = df_ls.rename(columns = {'tr_ld1':'tr'})

    #==============================
    #  Compute Individual Revenue
    #==============================

    #Compute revenue
    df_ls['ret'] = df_ls['position']*df_ls['pi']*df_ls['tr']
    
    #==============================
    #  Compute Aggregated Revenue
    #==============================
    
    df_profit = (df_ls
                 .groupby('eom')['ret'].sum()
                 .reset_index()
                 .rename(columns = {0: 'ret'})
                 )
    df_profit['cumret'] = (1 + df_profit['ret']).cumprod()
    
    return df_ls, df_profit


def build_mega_liq_factors(sp500_only = False, q1=None, q2=None):
    """
    Construct MegaCap and Liquidity Fama–French-style factors using dependent
    double sorts, and return a single DataFrame containing both factors merged
    by month.

    The function:
      1. Builds a monthly stock-level panel with size, liquidity, and returns.
      2. Optionally restricts the universe to S&P 500 constituents.
      3. Computes a MegaCap factor (MegaCap − Non-MegaCap) within liquidity bins.
      4. Computes a Liquidity factor (Mega-Liquid − Illiquid) within size bins.
      5. Merges both factor time series on the date column.

    Parameters
    ----------
    sp500_only : bool, default False
        If True, restricts the sample to stocks that are members of the S&P 500
        in a given month.

    q1 : list-like or None
        Quantile cutpoints for the outer sort (e.g., [0.9] for top-decile).
        If None, defaults to [0.9].

    q2 : list-like or None
        Quantile cutpoints for the inner sort (e.g., [0.9] for top-decile).
        If None, defaults to [0.9].

    Returns
    -------
    pandas.DataFrame
        Monthly factor returns with columns:
        ['eom', 'MegaLiq', 'MegaCap'], sorted by date.

    Notes
    -----
    This function assumes the following objects already exist in scope:
        df_me, df_dolvol, df_returns, df_sp500_ids

    It also relies on the helper functions defined within:
        assign_portfolio, _wavg, double_sort, compute_FF_factor
    """
    
    def assign_portfolio(df, sorting_variable, q, labels):
        """
        Assign portfolio labels based on quantile breakpoints of a sorting variable.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the sorting variable.
        
        sorting_variable : str
            Column name of the variable used for sorting.
        
        q : list-like
            Interior quantile cutpoints (e.g., [0.9] for a two-bin sort).
        
        labels : list-like
            Portfolio labels corresponding to the resulting quantile bins.
        
        Returns
        -------
        pandas.Series
            Categorical portfolio assignments aligned with df.index.
        
        Notes
        -----
        Uses pandas.qcut with explicit [0, q..., 1] quantile endpoints and
        drops duplicate bins if the data do not support all cutpoints.
        """
        quantiles = [0] + q + [1.0]
        
        pfs = pd.qcut(df[sorting_variable],
                     q = quantiles,
                     labels = labels, 
                     duplicates = 'drop') 
        
        return pfs

    def _wavg(x, ret_col, w_col=None):
        """
        Compute weighted average returns for a grouped DataFrame.
        
        Parameters
        ----------
        x : pandas.DataFrame
            Grouped subset of the data.
        
        ret_col : str
            Column name containing returns.
        
        w_col : str or None, default None
            Column name containing weights (e.g., market equity).
            If None, computes an equal-weighted mean return.
        
        Returns
        -------
        float
            Weighted (or equal-weighted) average return for the group.
        
        Notes
        -----
        Observations with missing returns, missing weights, or non-positive
        weights are excluded from the weighted calculation.
        """
        
        r = x[ret_col]
        if w_col is None:
            return r.mean()
        w = x[w_col]
        mask = r.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return np.average(r[mask], weights=w[mask])

    def double_sort(
        df,
        date_col="eom",
        id_col="id",
        ret_col="tr_ld1",
        sort1="me",
        pf1 = "PF_Mega",
        q1=[0.9],
        labels1=("Non-MegaCap", "MegaCap"),
        sort2="dolvol_126d",
        pf2 = "PF_Liq",
        q2=[0.9],
        labels2=("Illiquid", "Mega-Liquid"),
        independent=False,
        value_weighted=True,
        weight_col="me",
        dropna_sorts=True
    ):
        
        """
        Perform a two-way portfolio sort (independent or dependent) and compute
        portfolio returns.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Stock-level panel containing identifiers, returns, and sorting variables.
    
        date_col : str, default "eom"
            Time index column used for portfolio formation.
    
        id_col : str, default "id"
            Security identifier column (not directly used but kept for clarity).
    
        ret_col : str, default "tr_ld1"
            Column containing asset returns.
    
        sort1 : str
            Variable used for the outer (first) sort.
    
        pf1 : str
            Name of the portfolio column created by the outer sort.
    
        q1 : list-like
            Quantile cutpoints for the outer sort.
    
        labels1 : tuple
            Portfolio labels for the outer sort.
    
        sort2 : str
            Variable used for the inner (second) sort.
    
        pf2 : str
            Name of the portfolio column created by the inner sort.
    
        q2 : list-like
            Quantile cutpoints for the inner sort.
    
        labels2 : tuple
            Portfolio labels for the inner sort.
    
        independent : bool, default False
            If True, performs independent sorts.
            If False, performs dependent (conditional) sorts.
    
        value_weighted : bool, default True
            If True, computes value-weighted portfolio returns.
            If False, computes equal-weighted returns.
    
        weight_col : str, default "me"
            Column used as weights when value_weighted is True.
    
        dropna_sorts : bool, default True
            If True, drops observations with missing sorting variables.
    
        Returns
        -------
        df_out : pandas.DataFrame
            Original DataFrame with portfolio assignment columns added.
    
        port_ret : pandas.DataFrame
            Portfolio return table indexed by (date, pf1, pf2).
        """

        df_out = df.copy()

        # Optionally drop rows missing sorting variables
        if dropna_sorts:
            df_out = df_out[df_out[sort1].notna() & df_out[sort2].notna()]

        # --- sort 1 always by date ---
        df_out[pf1] = (
            df_out.groupby(date_col, group_keys=False) # +  include_groups=False
                  .apply(lambda g: assign_portfolio(g, sort1, q1, list(labels1)))
        )

        # --- sort 2 depends on independent vs dependent ---
        if independent:
            # independent: sort2 within date only
            df_out[pf2] = (
                df_out.groupby(date_col, group_keys=False) # + include_groups=False
                      .apply(lambda g: assign_portfolio(g, sort2, q2, list(labels2)))
            )
        else:
            # dependent: sort2 within (date, pf1)
            # (pf1 must exist first, so we sort within each pf1 bucket per date)
            df_out[pf2] = (
                df_out.groupby([date_col, pf1], group_keys=False) # + include_groups=False
                      .apply(lambda g: assign_portfolio(g, sort2, q2, list(labels2)))
            )

        # --- compute portfolio returns ---
        wcol = weight_col if value_weighted else None

        port_ret = (
            df_out.dropna(subset=[pf1, pf2, ret_col])
                  .groupby([date_col, pf1, pf2])
                  .apply(lambda x: _wavg(x, ret_col, wcol))
                  .rename("ret")
                  .reset_index()
        )

        return df_out, port_ret
    
    def compute_FF_factor(
        ret_df,
        date_col="eom",
        cond_col="PF_Liq",         # Outer sort column
        inner_col="PF_Mega",       # inner sort column
        ret_col="ret",
        inner_hi="MegaCap",
        inner_lo="Non-MegaCap",
        factor_name = "Mega"
    ):
        
        """
        Compute a Fama–French-style factor from double-sorted portfolio returns.
        
        The factor is constructed as the average, across outer-sort buckets, of
        the return spread between high and low inner-sort portfolios.
        
        Parameters
        ----------
        ret_df : pandas.DataFrame
            Portfolio return table produced by double_sort.
        
        date_col : str, default "eom"
            Time index column.
        
        cond_col : str
            Outer-sort portfolio column.
        
        inner_col : str
            Inner-sort portfolio column.
        
        ret_col : str, default "ret"
            Column containing portfolio returns.
        
        inner_hi : str
            Label of the high inner-sort portfolio.
        
        inner_lo : str
            Label of the low inner-sort portfolio.
        
        factor_name : str
            Name assigned to the resulting factor return column.
        
        Returns
        -------
        pandas.DataFrame
            Monthly factor return series with columns [date_col, factor_name].
        
        Notes
        -----
        This mirrors the standard Fama–French construction:
          factor_t = average_conditional( R_hi − R_lo ).
        """
        
        spreads_by_liq = (ret_df
                          .set_index(["eom", cond_col, inner_col])["ret"]
                          .unstack(inner_col) # Makes two columns per pf1
                          )
        
        # Per date: Spread of inner_col within cond_col
        avg_spread_by_liq = spreads_by_liq[inner_hi] - spreads_by_liq[inner_lo]
        
        # Per date: Average across cond_col buckets
        factor = (avg_spread_by_liq
                  .groupby("eom")
                  .mean()
                  .reset_index()
                  .rename(columns = {0:factor_name})
                  )

        return factor
    
    if q1 is None:
        q1 = [0.9]
    if q2 is None:
        q2 = [0.9]

    # --- build base panel ---
    df = (
        df_me
        .merge(df_dolvol, on=["id", "eom"], how="left")
        .merge(df_returns[["id", "eom", "tr_ld1"]], on=["id", "eom"], how="left")
        .merge(df_sp500_ids.assign(in_sp500=True), how="left", on=["id", "eom"])
        .assign(in_sp500=lambda d: d["in_sp500"].fillna(False))
    )

    if sp500_only:
        df = df.loc[df["in_sp500"]].drop(columns=["in_sp500"])

    # ------------------------------------------------------------------
    # 1) Outer Liquidity, inner MktCap  -> factor_Mega (MegaCap spread)
    # ------------------------------------------------------------------
    df_Mega, ret_Mega = double_sort(
        df,
        date_col="eom",
        id_col="id",
        ret_col="tr_ld1",
        sort1="dolvol_126d",
        pf1="PF_Liq",
        q1=q1,
        labels1=("Illiquid", "Mega-Liquid"),
        sort2="me",
        pf2="PF_Mega",
        q2=q2,
        labels2=("Non-MegaCap", "MegaCap"),
        independent=False,
        value_weighted=True,
    )

    factor_Mega = compute_FF_factor(
        ret_Mega,
        date_col="eom",
        cond_col="PF_Liq",
        inner_col="PF_Mega",
        ret_col="ret",
        inner_hi="MegaCap",
        inner_lo="Non-MegaCap",
        factor_name="MegaCap",
    )

    # ------------------------------------------------------------------
    # 2) Outer MktCap, inner Liquidity -> factor_Liq (MegaLiq spread)
    # ------------------------------------------------------------------
    df_Liq, ret_Liq = double_sort(
        df,
        date_col="eom",
        id_col="id",
        ret_col="tr_ld1",
        sort1="me",
        pf1="PF_Mega",
        q1=q1,
        labels1=("Non-MegaCap", "MegaCap"),
        sort2="dolvol_126d",
        pf2="PF_Liq",
        q2=q2,
        labels2=("Illiquid", "Mega-Liquid"),
        independent=False,
        value_weighted=True,
    )

    factor_Liq = compute_FF_factor(
        ret_Liq,
        date_col="eom",
        cond_col="PF_Mega",
        inner_col="PF_Liq",
        ret_col="ret",
        inner_hi="Mega-Liquid",
        inner_lo="Illiquid",
        factor_name="MegaLiq",
    )

    # --- single output: factor_Mega merged onto factor_Liq by eom ---
    out = (
        factor_Liq
        .merge(factor_Mega, on="eom", how="left")
        .sort_values("eom")
        .reset_index(drop=True)
    )

    return out


def hypothetical_TC(predictor, labels, target_col, est_univs, input_feat, hp_tuning, 
                    pi_max, volScaler, tcReguliser, 
                    df_wealth, df_kl):
    """
    Computes hypothetical transaction costs and net returns for multiple strategies.
    
    Returns:
        dict_dfs: Dictionary of detailed DataFrames keyed by label.
        dict_profits: Dictionary of aggregated profit DataFrames keyed by label.
    """
    dict_dfs = {}
    dict_profits = {}
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = pi_max,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tcReguliser, 
                        )
    
    # Scale factors for transaction costs
    tc_scalers = [1.0, 0.5, 0.1, 0.01]

    for i in range(len(predictor)):
        label = labels[i]
        
        # 1. Generate path and load data
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        try:
            with open(f"{path}Portfolios/{load_string}", "rb") as f:
                df = (pickle.load(f)['Strategy']
                      .drop(columns=['lambda', 'tc'], errors='ignore'))
        except FileNotFoundError:
            print(f"Warning: File not found for {label}")
            continue

        # 2. Merge auxiliary data
        # Baseline Wealth at BEGINNING of month
        df = df.merge(df_wealth.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                      [['eom_lead','wealth']],
                      left_on = ['eom'], right_on=['eom_lead'], how='left').drop(columns='eom_lead')
        # Baseline Lambda at BEGINNING of month
        df = df.merge(df_kl.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                      [['id','eom_lead','lambda']], left_on=['id', 'eom'],
                      right_on = ['id','eom_lead'], 
                      how = 'left').drop(columns='eom_lead')


        # 3. Compute Hypothetical Transaction Costs (Loop through scalers)
        for tc_scaler in tc_scalers:
            tc_col = f"tc_{tc_scaler}"
            ret_net_col = f"ret_net_{tc_scaler}"
            
            df = df.assign(**{
                tc_col: lambda x, s=tc_scaler: s * x['wealth'] * 0.5 * x['lambda'] * (x['pi'] - x['pi_g_tm1'])**2, # Caution: 0.5 due to JKMP22 legacy code.
                ret_net_col: lambda x, tc=tc_col: x['rev'] - x[tc]
            })

        # Store the granular dataframe
        dict_dfs[label] = df

        # 4. Aggregate results for df_profit
        df_profit = None
        for tc_scaler in tc_scalers:
            ret_net_col = f"ret_net_{tc_scaler}"
            cum_ret_col = f"cum_net_{tc_scaler}"
            
            df_add = (df
                      .groupby('eom', as_index=False)
                      .agg(**{ret_net_col: (ret_net_col, 'sum')})
                      .sort_values('eom')
                      .assign(**{cum_ret_col: lambda x: (1.0 + x[ret_net_col]).cumprod()}))

            # --- Logic to handle missing initial df_profit ---
            if df_profit is None:
                df_profit = df_add.copy()
            else:
                # For subsequent scalers, merge into the base
                df_profit = df_profit.merge(
                    df_add[['eom', ret_net_col, cum_ret_col]], 
                    on='eom', 
                    how='left'
                )
            
        dict_profits[label] = df_profit

    return dict_dfs, dict_profits

def compute_TC_from_Scratch(df_strat, df_wealth, df_returns, df_sp500_ids, tc_scalers, name):   
    """
    df_strat must contain ['id','eom','pi', 'tr'], where 'pi' is the 
    beginning of month portfolio weight.
    """
    
    # Expand the dataset to the entire universe (to get newcomers, leavers and stayers)
    df_universe = df_sp500_ids.loc[(df_sp500_ids['eom'] >= df_strat.eom.min())
                                    &
                                    (df_sp500_ids['eom'] <= df_strat.eom.max())]
    df_universe = df_universe.sort_values(by = ['id','eom'])
    
    df_strat = df_universe.merge(df_strat, on = ['id','eom'], how = 'left')
    df_strat = df_strat.fillna(0)
        
    
    # Merge g^w, lambda and wealth
    df_strat = (df_strat
                .merge(df_wealth, on = ['eom'], how = 'left')
                .merge(df_kl, on = ['id','eom'], how = 'left').drop("eom_lead",axis = 1)
                ).sort_values(by = ['id','eom'])
    
    # Compute g (coefficient for drifted portfolio weight)
    df_strat = df_strat.assign(g = lambda df: (1+df['tr']) / (1+df['mu']))
    
    # Calculate drifted portfolio weight (important! sorted by 'id' and 'eom'. Otherwise shift-operator yields incorrect results)
    df_strat['pi_g_tm1'] = df_strat.groupby('id').apply(
        lambda x: (x['pi'] * x['g']).shift(1)
    ).reset_index(level=0, drop=True)

    # No tc in first period for simplicity
    initial_date = df_strat['eom'].min()
    mask = df_strat['eom'] == initial_date
    df_strat.loc[mask, 'pi_g_tm1'] = df_strat.loc[mask, 'pi']
    
    
    # Compute revenue
    df_strat = df_strat.assign(rev = lambda df: df['tr']*df['pi'])
    
    # Compute TC & net return
    for tc_scaler in tc_scalers:
        tc_col = f"tc_{tc_scaler}"
        ret_net_col = f"ret_net_{tc_scaler}"
        
        df_strat = df_strat.assign(**{
            tc_col: lambda x, s=tc_scaler: s * x['wealth'] * 0.5 * x['lambda'] * (x['pi'] - x['pi_g_tm1'])**2, # Caution: 0.5 due to JKMP22 legacy code.
            ret_net_col: lambda x, tc=tc_col: x['rev'] - x[tc]
        })
        
    # Compute Profit
    net_ret_cols = [f"ret_net_{s}" for s in tc_scalers]
    df_profit = df_strat.groupby('eom')[net_ret_cols].sum().reset_index()
    
    # Get cumulative Profit
    for tc_scaler in tc_scalers:
        ret_col = f"ret_net_{tc_scaler}"
        # Creating a unique name for the cumulative column based on strategy and scaler
        cum_col = f"cumret_net_{tc_scaler}_{name}"
        
        df_profit[cum_col] = (1 + df_profit[ret_col]).cumprod()
        
        # Optional: Rename the raw return columns to include the Strategy name 
        # This prevents collisions when merging with Momentum later
        df_profit = df_profit.rename(columns={ret_col: f"{ret_col}_{name}"})
        
    return df_strat, df_profit

#%% Function: Load Portfolios

def value_to_token(v):
    """Convert Python value to a compact string token for filenames."""
    if isinstance(v, bool):
        return str(v).lower()          # True -> "true", False -> "false"
    if v is None:
        return "None"
    if isinstance(v, float):
        # Example: 0.15 -> "015", 1.0 -> "10"
        s = f"{v:.3f}".rstrip("0").rstrip(".")  # "0.15"
        return s.replace(".", "")               # "015"
    return str(v)

def settings_string(settings: dict) -> str:
    # Preserve insertion order (Python 3.7+)
    parts = [f"{k}={value_to_token(v)}" for k, v in settings.items()]
    return "_".join(parts)

def plot_string(settings: dict) -> str:
    # Preserve insertion order (Python 3.7+)
    parts = [f"{k}={value_to_token(v)}" for k, v in settings.items() 
             if k in ['flatMaxPiVal', 'volScaler', 'tcScaler']]
    return "_".join(parts)


def get_strats(
    *, # accept only keyword arguments
    path: str,
    df_wealth,
    df_kl,
    tc_scalers: List[float] = (1.0, 0.5, 0.1, 0.01),

    predictor: List[str] = (
        'XGBRegHPlenient',
        'TransformerSet_Dropout010',
        'IPCA',
        'RFF',
    ),
    labels: List[str] = (
        'XGB',
        'Transformer',
        'IPCA',
        'RFF',
    ),
    target_col: List[str] = (
        'LevelTrMsp500Target',
        'LevelTrMsp500Target',
        'LevelTrMsp500Target',
        'LevelTrMsp500Target',
    ),
    est_univs: List[str] = (
        'SP500UniverseFL',
        'SP500UniverseFL',
        'CRSPUniverse',
        'SP500UniverseFL',
    ),
    input_feat: List[str] = (
        'RankFeatures',
        'RankFeatures',
        'ZscoreFeatures',
        'ZscoreFeatures',
    ),
    hp_tuning: str = "RollingWindow_win120_val12_test12",

    flatMaxPiVal: float = 0.15,
    volScaler: float = 1.0,
    tcReguliser: float = 1.0,

    includeRF: bool = False,
    flatMaxPi: bool = True,
    Wmax=None,
    Wmin=None,
) -> Dict[str, Any]:

    if not (len(predictor) == len(labels) == len(target_col) == len(est_univs) == len(input_feat)):
        raise ValueError("All model configuration lists must have the same length.")

    # -------------------------
    # Hypothetical TC (always Act as Large)
    # -------------------------
    hypo_strats, hypo_profits = hypothetical_TC(
        predictor=list(predictor),
        labels=list(labels),
        target_col=list(target_col),
        est_univs=list(est_univs),
        input_feat=list(input_feat),
        hp_tuning=hp_tuning,
        pi_max=flatMaxPiVal,
        volScaler=volScaler,
        tcReguliser=tcReguliser,
        df_wealth=df_wealth,
        df_kl=df_kl,
    )

    out = {
        "hypo": {
            label: {
                "Profit": hypo_profits.get(label),
                "Strategy": hypo_strats.get(label),
            }
            for label in labels
        },
        "actual": {label: {} for label in labels},
    }

    # -------------------------
    # Actual TC (Act as = Pay as)
    # -------------------------
    for tc_scaler in tc_scalers:
        run_settings = dict(
            includeRF=includeRF,
            flatMaxPi=flatMaxPi,
            flatMaxPiVal=flatMaxPiVal,
            Wmax=Wmax,
            Wmin=Wmin,
            volScaler=volScaler,
            tcScaler=tc_scaler,
        )

        for i, label in enumerate(labels):
            load_string = (
                f"{settings_string(run_settings)}_"
                f"{predictor[i]}_"
                f"{target_col[i]}_"
                f"{est_univs[i]}_"
                f"{input_feat[i]}_"
                f"{hp_tuning}.pkl"
            )

            with open(f"{path}Portfolios/{load_string}", "rb") as f:
                data = pickle.load(f)

            out["actual"][label][tc_scaler] = {
                "Profit": data.get("Profit"),
                "Strategy": data.get("Strategy"),
                "run_settings": run_settings,
            }

    return out


#%% Long-Short Portfolio on CRSP

# ===========
# Load Data
# ===========

# ---- Predictor ----
predictors = ['XGBReg',
             'TransformerSet_Dropout005',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrTarget']

# ---- Est. Universe ----
est_univs = ['CRSPUniverse',
             'CRSPUniverse',
             'CRSPUniverse',
             'CRSPUniverse']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Load Return Predictions ----
predictions = []
for i, predictor in enumerate(predictors):
    # Recall that at date 'eom' the prediction is for 'eom' + 1 
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    
    # If operating on SP500 Universe only
    # df = df.merge(df_sp500_ids.assign(in_sp500 = True), on = ['id', 'eom'], how = 'left')
    # df = df.dropna().drop(columns = 'in_sp500')
    
    predictions.append([df, labels[i]])
    
# ---- Compute Long-Short Portfolios ----
all_strategy_profits = []
sharpe_results = []

for df_pred, label in predictions:
    # 1. Identify the prediction column dynamically
    # It's the column that isn't 'id' or 'eom'
    pred_col = [col for col in df_pred.columns if col not in ['id', 'eom']][0]
    
    print(f"Processing {label} using column: {pred_col}")
    
    # Long-Short Portfolio
    df_ls, df_profit = long_short_portfolio(
        df=df_pred,
        prediction_col=pred_col,
        df_returns=df_returns, 
        df_me=df_me,
        long_cutoff=0.9, 
        short_cutoff=0.1,
        value_weighted=False, 
        long_only = False
    )
    
    # Add the label to the profit dataframe for identification
    df_profit['model'] = label
    all_strategy_profits.append([df_ls,df_profit])
    
    # Sharpe Ratio
    mu, sigma = meanRet_varRet(df_profit, 'ret')
    sharpe = SharpeRatio(df_profit, risk_free, 'ret')
    ir = InformationRatio(df_profit, df_spy, 'ret', 'ret', risk_free)
    
    sharpe_results.append({
        'Model': label,
        'mu': mu,
        'sigma': sigma,
        'SR': sharpe,
        'IR': ir
        
    })
    
df_metrics = (pd.DataFrame(sharpe_results)
              .rename(columns = {'mu':'$\mu$', 'sigma':'$\sigma$'})
              )


print(df_metrics.to_latex(index=False, escape=False, float_format="%.3f"))

#%% Long-Short Portfolio on S&P 500

# ---- Predictor ----
predictors = [        'XGBRegHPlenient',
        'TransformerSet_Dropout010',
        'IPCA',
        'RFF',]

# ---- Label ----
labels = [        'XGB',
        'Transformer',
        'IPCA',
        'RFF']

# ---- Target Type ----
target_col = [        'LevelTrMsp500Target',
        'LevelTrMsp500Target',
        'LevelTrMsp500Target',
        'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = [        'SP500UniverseFL',
        'SP500UniverseFL',
        'CRSPUniverse',
        'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Load Return Predictions ----
predictions = []
for i, predictor in enumerate(predictors):
    # Recall that at date 'eom' the prediction is for 'eom' + 1 
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    
    # Operating on SP500 Universe only
    df = df.merge(df_sp500_ids.assign(in_sp500 = True), on = ['id', 'eom'], how = 'left')
    df = df.dropna().drop(columns = 'in_sp500')
    
    predictions.append([df, labels[i]])
    
# ---- Compute Long-Short Portfolios ----
all_strategy_profits = []
sharpe_results = []

for df_pred, label in predictions:
    # 1. Identify the prediction column dynamically
    # It's the column that isn't 'id' or 'eom'
    pred_col = [col for col in df_pred.columns if col not in ['id', 'eom']][0]
    
    print(f"Processing {label} using column: {pred_col}")
    
    # Long-Short Portfolio
    df_ls, df_profit = long_short_portfolio(
        df=df_pred,
        prediction_col=pred_col,
        df_returns=df_returns, 
        df_me=df_me,
        long_cutoff=0.9, 
        short_cutoff=0.1,
        value_weighted=False, 
        long_only = False
    )
    
    # Add the label to the profit dataframe for identification
    df_profit['model'] = label
    all_strategy_profits.append([df_ls,df_profit])
    
    # Sharpe Ratio
    mu, sigma = meanRet_varRet(df_profit, 'ret')
    sharpe = SharpeRatio(df_profit, risk_free, 'ret')
    ir = InformationRatio(df_profit, df_spy, 'ret', 'ret', risk_free)
    
    sharpe_results.append({
        'Model': label,
        'mu': mu,
        'sigma': sigma,
        'SR': sharpe,
        #'IR': ir
        
    })
    
df_metrics = (pd.DataFrame(sharpe_results)
              .rename(columns = {'mu':'$\mu$', 'sigma':'$\sigma$'})
              )


print(df_metrics.to_latex(index=False, escape=False, float_format="%.3f"))




#%% Plot Cumulative Return Ratio

"""
Plot the Ratio of 
    cumulative strategy returns 
            over 
    cumualtive S&P 500 return
"""

# Broadcom: 93002
# NVDA: 86580 
# AAPL: 14593

portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 1.0,     # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        )

# Which Portfolio to Plot
Act_as_equalTo_Pay_as = False 

# Which transaction cost regime to plot
tc_scaler = 0.01


# ---- Save Figure ----
Save_Figure = True
plot_filename = path + "Plots/CumRet_Ratio_ActasLarge.pdf"


# ---- Y-Axis ----
y_low, y_high = 0.9, 1.8
tick_range = np.round([y_low + 0.1*i for i in range(int((y_high - y_low)*10+1))],2)
fontsize = 18

# ---- Set the color ----
colors = {
    "XGB":          "#1f77b4",  # XGBoost
    "Transformer":  "#658A0B",  # Transformer
    "IPCA":         "#967969",  # IPCA
    "RFF":          "#d62728",  # RFF
}

# ---- Figure Dimensions ----
fig, ax = plt.subplots(figsize=(10, 6))

# ---- Load CumRets ----
for strat in ['XGB', 'Transformer', 'IPCA', 'RFF']:
    # ---- Act as and Pay as tc_scaler ----
    if Act_as_equalTo_Pay_as:
        ycol = "cumret_net"
        df_profit = portfolios["actual"][strat][tc_scaler]['Profit'][["eom", ycol]]
    
    # ---- Act as Large, Pay as tc_scaler ----
    else:
        # Get column name
        ycol = f"cum_net_{tc_scaler}"
        # Extract Profit of strat and tc combo
        df_profit = portfolios["hypo"][strat]['Profit'][["eom", ycol]]
        
    # Merge S&P 500 CumRet
    df_profit = df_profit.merge(df_spy[['eom','cumulative_return']], on = 'eom', how = 'left')
    # Compute CumRet Ratio 
    df_profit['cumret_ratio'] = df_profit[ycol] / df_profit['cumulative_return']
    ax.plot(
        df_profit['eom'],
        df_profit['cumret_ratio'],
        label=strat,
        color = colors[strat], 
        alpha=0.9,
        linewidth=1.5,
        zorder=1
    )

# --- Labels ---
ax.set_ylabel("Cumulative Return Ratio", fontsize=fontsize)

# --- Log Scale ---
#ax.set_yscale("log")
# Force the y-axis to use decimal labels instead of scientific notation
#ax.yaxis.set_major_formatter(ScalarFormatter())

# Ensure the formatter doesn't revert to scientific notation for small/large numbers
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

# ---- Grid Lines ----
ax.grid(visible=True, which='major', color='gray', linestyle='-', alpha=0.4, zorder=0)
ax.grid(visible=True, which='minor', color='gray', linestyle=':', alpha=0.2, zorder=0)

# --- Year ticks: every year, vertical labels ---
all_dates = df_spy['eom']
ax.set_xlim(all_dates.min(), all_dates.max())
ax.xaxis.set_major_locator(mdates.YearLocator(1))    # tick every year
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))    # show as 2020, 2021, ...
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")    # vertical

# --- Tick font sizes ---
ax.tick_params(axis="both", which="major", labelsize=fontsize)

#Set Ticks 
ax.set_yticks(tick_range) 
ax.set_ylim(y_low, y_high)

# --- Legend font size ---
ax.legend(fontsize=fontsize)

# --- Save Plot ---
plt.tight_layout()
if Save_Figure:
    plt.savefig(plot_filename, dpi=300)

# ---- Display Plot ----
plt.show()


#%% Ridge Plot

# ================
#     Setting
# ================

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF',
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              ]

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"


run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 0.01, # Ridge regularisation always as if TC large. This here gives the transaction costs paid.
                    )

# Container to Store strats
dict_strats_Ridge   = {}
dict_profits_Ridge = {}


for i in range(len(predictor)):
    load_string = ("Ridge_"
                    f"{settings_string(run_settings)}_" 
                   f"{predictor[i]}_" 
                   f"{target_col[i]}_" 
                   f"{est_univs[i]}_" 
                   f"{input_feat[i]}_"
                   f"{hp_tuning}.pkl")
    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        obj = pickle.load(f)
    
    print(obj['model_name'])
    dict_strats_Ridge[labels[i]]    = obj['Strategy']
    dict_profits_Ridge[labels[i]]   = obj['Profit']

colors = {
    'XGB': "#1f77b4",  # XGBoost
    'Transformer':"#658A0B",  # Transformer
    'IPCA':"#967969",  # IPCA
    'RFF':"#d62728",  # RFF
}

# ---- Figure Dimensions ----
fig, ax = plt.subplots(figsize=(10, 6))
plot_filename = path + "Plots/CumRetRidge_Ratio_ActasLarge_PayAsTiny.pdf"
Save_Figure = False
fontsize = 18

# ---- Load CumRets ----
for key, df_profit in dict_profits_Ridge.items():
    ycol = 'cumret_net'
    df_profit = df_profit.merge(df_spy[['eom','ret','cumulative_return']], on = 'eom', how = 'left')
    df_profit['cumret_Ratio'] = df_profit[ycol]/df_profit['cumulative_return']
    
    # Can do the t-test on the hit-ratio of the cumret_Ratio
    # print(key, (df_profit['cumret_Ratio'].diff() >0).sum()/len(df_profit))
    
    ax.plot(
        df_profit['eom'],
        df_profit['cumret_Ratio'],
        label=key,
        color = colors[key], 
        alpha=0.9,
        linewidth=1.5,
        zorder=1
    )

# --- Labels ---
ax.set_ylabel("Ratio CumRet Strat over CumRet S&P 500", fontsize=fontsize)

# Ensure the formatter doesn't revert to scientific notation for small/large numbers
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

# ---- Grid Lines ----
ax.grid(visible=True, which='major', color='gray', linestyle='-', alpha=0.4, zorder=0)
ax.grid(visible=True, which='minor', color='gray', linestyle=':', alpha=0.2, zorder=0)

# --- Year ticks: every year, vertical labels ---
all_dates = df_spy['eom']
ax.set_xlim(all_dates.min(), all_dates.max())
ax.xaxis.set_major_locator(mdates.YearLocator(1))    # tick every year
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))    # show as 2020, 2021, ...
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")    # vertical

# --- Tick font sizes ---
ax.tick_params(axis="both", which="major", labelsize=fontsize)

# --- Legend font size ---
ax.legend(fontsize=fontsize)

# --- Save Plot ---
plt.tight_layout()
if Save_Figure:
    plt.savefig(plot_filename, dpi=300)

# ---- Display Plot ----
plt.show()


#%% Table Summary Performance Statistics

""" 
Computes the performance metrics and outputs the results as a Copy&Paste-ready
LaTeX Table
"""

#Act as TC parameter
tcReguliser = 0.5
# Load Portfolios
portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 0.1,       # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        tcReguliser= tcReguliser,
                        )

# Which Portfolios to display
Act_as_equalTo_Pay_as = False

# Strats to display
strats = ["XGB", "Transformer", "IPCA", "RFF"]

# Naming in Table
tc_map = {1.0: "Large", 0.5: "Med", 0.1: "Small", 0.01: "Tiny"}

# Master container storing results
all_results = {strat: {} for strat in portfolios["hypo"].keys()}

# Calculate Benchmark (only needs to run once)
mu_bench, sigma_bench, = meanRet_varRet(df_spy, 'ret')
Sharpe_bench = SharpeRatio(df_spy, risk_free, return_col='ret')
drawdown_bench, _, _ = MaxDrawdown(df_spy, df_spy, 'ret', 'ret')

#---- Compute Table of Performance Metrics for Portfolios ---
# Loop over Strats
for tc_scaler in [1.0, 0.5, 0.1, 0.01]:
    
    perform_dict = {}
    for strat in strats:
        # Load Portfolios
        if Act_as_equalTo_Pay_as: # Act as and Pay as tc_scaler
            ret_col_strat = "ret_net"
            df_profit   = portfolios["actual"][strat][tc_scaler]['Profit']
            df_strategy = portfolios["actual"][strat][tc_scaler]['Strategy']
        
        else: #  Act as Large, Pay as tc_scaler
            # Get column name
            ret_col_strat = f"ret_net_{tc_scaler}"
            # Extract Profit of strat and tc combo
            df_profit   = portfolios["hypo"][strat]['Profit']
            df_strategy = portfolios["hypo"][strat]['Strategy']
        
        
        mu_s, sigma_s = meanRet_varRet(df_profit, ret_col_strat)
        sharpe_s = SharpeRatio(df_profit, risk_free, return_col=ret_col_strat)
        ir = InformationRatio(df_profit, df_spy, ret_col_strat, 'ret', risk_free)
        to, _ = Turnover(df_strategy, df_wealth)
        dd, _, _ = MaxDrawdown(df_profit, df_spy, ret_col_strat, 'ret')
        down_cap, _, _, _ = CaptureRatio(df_profit, df_spy, ret_col_strat, 'ret')
        
        alpha_ann, p_value = capm_alpha(df_profit, df_spy, ret_col_strat, 'ret', risk_free)

        # Store in master dictionary
        all_results[strat][tc_map[tc_scaler]] = {
            'mu': mu_s, 'sigma': sigma_s, 'sharpe': sharpe_s,
            'turnover': to, 'ir': ir, 'dd': dd, 'down_cap': down_cap,
            'alpha': alpha_ann, 
            'alpha_p': p_value
        }

# ==========================================
# 3. Manual LaTeX Table Construction
# ==========================================
status_str = "Net" 

latex_str = f"""\\begin{{table}}[htpb]
\\centering
\\caption{{Summary Statistics {status_str} Returns}}
\\label{{Table:SummaryStats_{status_str}Returns}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{llcccccccc}}
\\toprule
Act as & Pay as & $\\mu$ & $\\sigma$ & SR & TO & IR & MaxD & DCap & $\\alpha$ \\\\
\\midrule
"""

latex_str += (
    f"$\\bullet$ & $\\bullet$ & {mu_bench:.3f} & {sigma_bench:.3f} & {Sharpe_bench:.3f} & "
    f"$\\bullet$ & 0.000 & {drawdown_bench:.3f} & 1.000 & 0.000 \\\\\n"
)
latex_str += "\\bottomrule\n"

# Model Blocks
for strat in strats:
    latex_str += "%" + "="*50 + "\n"
    latex_str += "\\toprule\n"
    latex_str += f"\\multicolumn{{10}}{{c}}{{\\textbf{{{strat}}}}} \\\\\n"
    
    # Iterate through TC statuses in specific order
    for tc_status in ["Large", "Med", "Small", "Tiny"]:
        if tc_status in all_results[strat]:
            res = all_results[strat][tc_status]
    
            # Display strat: High should be Large
            if Act_as_equalTo_Pay_as:
                act_as = tc_status 
            else:
                act_as = tc_map[tcReguliser]
    
            # Alpha with stars
            alpha_str = f"{res['alpha']:.3f}{star_notation(res['alpha_p'])}"
    
            row = (
                f"{act_as} & {tc_status} & {res['mu']:.3f} & {res['sigma']:.3f} & "
                f"{res['sharpe']:.3f} & {res['turnover']:.3f} & {res['ir']:.3f} & "
                f"{res['dd']:.3f} & {res['down_cap']:.3f} & {alpha_str} \\\\\n"
            )
            latex_str += row

    latex_str += "\\bottomrule\n"

latex_str += r"""\end{tabular}
\end{threeparttable}
\end{table}"""

print(latex_str)

#%% Portfolio Weights deviation

# Load Portfolios
portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 1.0,       # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        )

df_strat = portfolios["hypo"]["IPCA"]["Strategy"]

# Merge me of S&P 500 stocks at begin of month
df_strat = df_strat.merge((df_me
                           .assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                           .rename(columns = {'me':'me_lag'})
                           .drop('eom',axis=1)),
                          left_on = ['id','eom'],
                          right_on = ['id','eom_lead'],
                          how = 'left').drop('eom_lead',axis = 1)

# Get S&P 500 Portfolio Weight
df_strat['pi_sp500'] = df_strat.groupby('eom')['me_lag'].transform(lambda x: x/x.sum())
df_strat = df_strat.drop('me_lag',axis = 1)

df_strat['weight_dif'] = np.abs(df_strat['pi']-df_strat['pi_sp500'])













#%% Panel Regression Lambda

"""
Conditional on trading and controlling for time effects, XGBoost and RFF 
rebalance positions more aggressively in high–dollar-volume stocks than IPCA 
and the Transformer. This liquidity-sensitive trading behavior leads to 
significantly lower realized transaction costs. This is even causal. Time FE 
required as dolvol has a trend (decreasing) so that later time periods would
have a higher weight.

a linear-linear regression was numerically ill-conditioned. This is the case
as e.g. a one-unit increase in λ corresponds to destroying market liquidity entirely. 
That’s nonsense economically. The log helps to spread the tiny values more out
so that no tiny nudges in either dependent or independent variable have
large effects on the regression.

I use the square because this gives more (relative) weight to large deviations
and therefore pronounces aggressive trades more. Moreover, the TC are quadradic
in this way

Why I don't use a model by model regression:

XGB/RFF:
    Trade a small set of liquid winners (NVDA, AAPL)
    Often hit constraints (15% cap, volatility)
    Once they trade those names, they don’t need to scale trade size much with liquidity

IPCA:
    Trades a much broader cross-section
    Must adapt trade size within each month depending on liquidity
    Therefore shows a steeper conditional elasticity inside its own traded set

In Panel regressions with interactions:
→ XGB/RFF show stronger overall liquidity sorting. Each strategy has the
same time fixed effect

In strategy-by-strategy intensive regressions:
→ IPCA shows stronger within-strategy liquidity elasticity. Each strategy
has its own time fixed effects.

Individual Models: The model asks, "Relative to the average trading intensity of this specific strategy in month t, how does λ affect trading?"

Stacked Model: The model asks, "Relative to the average trading intensity of all strategies combined in month t, how does λ affect trading?"
"""

results_list = []

for tc_val in [1.0, 0.5, 0.1, 0.01]:

    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = 0.15,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = 1.0, 
                        tcScaler     = tc_val, 
                        )
    
    # ---- Predictor ----
    predictor = ['XGBRegHPlenient',
                 'TransformerSet_Dropout010',
                 'IPCA',
                 'RFF']
    
    # ---- Label ----
    labels = ['XGBoost', 
              'Transformer',
              'IPCA',
              'RFF']
    
    # ---- Target Type ----
    target_col = ['LevelTrMsp500Target', 
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target']
    
    # ---- Est. Universe ----
    est_univs = ['SP500UniverseFL',
                 'SP500UniverseFL',
                 'CRSPUniverse',
                 'SP500UniverseFL']
    
    # ---- Input Features ----
    input_feat = ['RankFeatures',
                  'RankFeatures',
                  'ZscoreFeatures',
                  'ZscoreFeatures']
    
    # ---- HP Tuning ----
    hp_tuning = "RollingWindow_win120_val12_test12"
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f)['Strategy'], labels[i]])
            
    
    all_data = []
    
    for df_strat, label in strats:
        # 1. Prepare variables for the regression
        # (Using a small constant for logs to avoid -inf if values are 0)
        df_temp = df_strat.copy()
        
        # Omit stocks that leave the trading universe (mechanical relationship with lambda)
        df_temp = df_temp.loc[df_temp['pi'] > 0]
        
        df_temp['dep_var'] = np.log((df_temp['pi'] - df_temp['pi_g_tm1'])**2)
        df_temp['log_lambda'] = np.log(df_temp['lambda'])
        df_temp['strategy'] = label  # To identify the group in the stacked model
        
        all_data.append(df_temp[['eom', 'dep_var', 'log_lambda', 'strategy']])
    
    # 2. Combine all strategies into one "stacked" DataFrame
    df_stacked = pd.concat(all_data)
    
    # 3. Run the regression
    # We include 'strategy' as a categorical fixed effect (alpha_t/label)
    # and interact log_lambda with strategy to get a beta for each label.
    # The "- 1" in the formula removes the global intercept to show individual strategy alphas.
    
    model = smf.ols(
        'dep_var ~ C(strategy):log_lambda + C(eom)',
        data=df_stacked
    ).fit(cov_type='HC1')
    
    print(model.summary())
    
    # Extract coefficients and confidence intervals
    conf_int = model.conf_int()
    params = model.params
    
    # Filter for the interaction terms (C(strategy)[...]:log_lambda)
    # This ignores the eom fixed effects
    for label in labels:
        coeff_name = f'C(strategy)[{label}]:log_lambda'
        if coeff_name in params.index:
            results_list.append({
                'tc_val': tc_val,
                'strategy': label,
                'beta': params[coeff_name],
                'lower': conf_int.loc[coeff_name, 0],
                'upper': conf_int.loc[coeff_name, 1]
            })

# Final DataFrame for plotting
df_ci = pd.DataFrame(results_list)
df_ci['strategy'] = df_ci['strategy'].replace({
    'XGBoost': 'XGB',
    'Transformer': 'TF'
})

# 1. Setup a single plot
fig, ax = plt.subplots(figsize=(10, 6))

tc_values = [1.0, 0.5, 0.1, 0.01]
limits_map = {
    1.0: (-1.6, -1.0),
    0.5: (-1.1, -0.4),
    0.1: (-0.2, 0.66),
    0.01: (0.88, 1.9)
}
desired_order = ['XGB', 'TF', 'IPCA', 'RFF']

# Colors for Plots
color_map = {
    'XGB': "#1f77b4",
    'TF': "#658A0B",
    'IPCA': "#967969",
    'RFF': "#d62728",
    'Market Oracle': "#AD9721"
}

# Iterate through each TC value and create a standalone figure for each
for tc in tc_values:
    # 1. Setup individual figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df_ci[df_ci['tc_val'] == tc].copy()
    
    # Ensure categorical order for the y-axis
    subset['strategy'] = pd.Categorical(subset['strategy'], categories=desired_order[::-1], ordered=True)
    subset = subset.sort_values('strategy')
    
    # 2. Plotting
    for _, row in subset.iterrows():
        strat_label = row['strategy']
        strat_color = color_map.get(strat_label, "black")
        
        error_low = row['beta'] - row['lower']
        error_high = row['upper'] - row['beta']
        
        ax.errorbar(
            row['beta'], strat_label, 
            xerr=[[error_low], [error_high]], 
            fmt='o', capsize=6, 
            color=strat_color, 
            markersize=9, 
            linewidth=2.5
        )
    
    # Tick Size
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 3. Individual Formatting
    if tc in limits_map:
        ax.set_xlim(limits_map[tc])
    
    # ax.set_title(f'Sensitivity of Trading Variance: TC Scale {tc}', fontweight='bold', fontsize=14)
    ax.set_xlabel('$\log (\lambda)$', fontsize=18)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(path + f"Plots/Log_Lambda_tc_{str(tc).replace(".", "")}.pdf")
    # This will display the current plot before moving to the next iteration
    plt.show()

#%% Transaction Costs: Averages

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF', 
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              ]

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Settings ----

tc_values       = [1.0, 0.5, 0.1, 0.01]
flatMaxPiVal    = 0.15
volScaler       = 1.0

# --- Objects for Storing ---
all_results = {label: {} for label in labels}
tc_map          = {1.0: "TC: High", 0.5: "TC: Med", 0.1: "TC: Low", 0.01: "TC: Tiny"}

for tc_scaler in tc_values:
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = flatMaxPiVal,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tc_scaler, 
                        )
    

    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        if predictor[i] != 'MarketOracle':
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}_" 
                           f"{target_col[i]}_" 
                           f"{est_univs[i]}_" 
                           f"{input_feat[i]}_"
                           f"{hp_tuning}.pkl")
        else:
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}"
                           ".pkl")
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f)['Strategy'], labels[i]]) # 'Strategy' for Portfolio Weights


    
    for item in strats:
        df_strat    = item[0]
        label       = item[1]
        
        all_results[label][tc_map[tc_scaler]] = df_strat['tc'].sum()


all_values = [val for model_dict in all_results.values()
                  for val in model_dict.values()]

global_max = max(all_values)


all_results = {key: {tc: val/global_max for tc, val in inner_dict.items()}
               for key, inner_dict in all_results.items()
               }

df = pd.DataFrame(all_results)
latex_table = df.to_latex(
    index=True, 
    caption="Normalized Transaction Costs across Regimes",
    label="Table:TC_Summary",
    float_format="%.3f",
    column_format="lcccc"
)

print(latex_table)    
#%% Variance Return Predictions
"""
Cross-sectional Variance of Return Predictions for each predictor.

Low Variance = low scale = TC more important in objective
"""
# ---- Predictor ----
predictors = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

predictions = []

for i, predictor in enumerate(predictors):
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{trading_start.strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    predictions.append([df, labels[i]])
    
variances   = []
for i, item in enumerate(predictions):
    # Unpack
    df      = item[0]
    label   = item[1]
    
    # Get Name of Prediction column
    ret_col = [col for col in df.columns if col not in ['eom', 'id']][0]
    # Compute cross-sectional variance for each date
    df_var = df.groupby('eom')[ret_col].var().rename(f"var_{labels[i]}")

    variances.append(df_var)
    
# Drop Transformer as it always has the highest variance and distorts results
df_variances = pd.concat(variances, axis = 1).drop(columns = "var_Transformer")
row_sum = df_variances.sum(axis=1)
df_variances = df_variances.div(row_sum, axis=0)

# ---- Plot ----

# Rename columns to remove 'var_' prefix for the legend
df_variances.columns = [col.replace('var_', '') for col in df_variances.columns]

# Color map keys to match the new names
color_map = {
    "XGBoost": "#1f77b4",
    "Transformer": "#658A0B",
    "IPCA": "#967969",
    "RFF": "#d62728",
    "Market Oracle": "#AD9721"
}

current_colors = [color_map.get(col, "#333333") for col in df_variances.columns]

# --- FORCE datetime index ---
dfv = df_variances.copy()
dfv.index = pd.to_datetime(dfv.index)
dfv = dfv.sort_index()

# --- Build x as actual datetimes ---
x = dfv.index.to_pydatetime()
cols = dfv.columns

fig, ax = plt.subplots(figsize=(12, 7))

# stackplot needs one array per series
ys = [dfv[c].to_numpy() for c in cols]

ax.stackplot(
    x,
    ys,
    labels=cols,
    colors=current_colors,
    alpha=0.85
)

# --- Force yearly ticks (use YearLocator + DateFormatter) ---
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xlabel(None)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylabel("Share of Total Variance", fontsize=18)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(cols),
    frameon=False,
    fontsize=18
)

ax.set_xlim(dfv.index.min(), dfv.index.max())
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(path + "Plots/Variances_RetPreds.pdf", dpi=300, bbox_inches='tight')
plt.show()
#%% Portfolio Visualisation
"""
Visualise the holdings of AAPL for XGBoost in tc = 1.0.

Probably best to display the percentile of tr and of retpred to
get the cross-sectional comparison
"""
# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 1.0, 
                    )
#tick_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
fontsize = 18
Save_Figure = False
Net = True
stock_id = 86580 #AAPL: 14593, NVDA: 86580

# ---- Predictor ----
predictor = ['XGBRegHPlenient']

# ---- Label ----
labels = ['XGB']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# Container to Store strats
strats = []

for i in range(len(predictor)):
    if predictor[i] != 'MarketOracle':
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
    else:
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}"
                       ".pkl")
    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        strats.append([pickle.load(f)['Strategy'], labels[i]]) # 'Strategy' for Portfolio Weights
        
        
for item in strats:
    df = item[0]
    label = item[1]
    df = df[['id','eom', 'pi', 'pi_g_tm1', 'tr', df.columns[-1]]] # Last column is the return prediction
    df = df.rename(columns = {'tr':'Real. Ret', df.columns[-1]:'Pred. Ret'})
    df = df[df['id'] == stock_id]
   
    df = df.sort_values('eom')
    
    # --- Plot 1: pi and pi_g_tm1 (Single Axis) ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['eom'], df['pi'], label='pi', marker='o')
    plt.plot(df['eom'], df['pi_g_tm1'], label='pi_g_tm1', linestyle='--', marker='x')
    
    plt.title('Comparison of PI and PI_G_TM1')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- Plot 2: tr and ret_pred_Levelmsp500 (Dual Axis) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Primary Y-Axis (Left) for 'Real. Ret'
    color_tr = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TR (Total Return)', color=color_tr)
    ax1.plot(df['eom'], df['Real. Ret'], color=color_tr, label='Real. Ret', marker='s')
    ax1.tick_params(axis='y', labelcolor=color_tr)
    
    # Secondary Y-Axis (Right) for 'Pred. Ret'
    ax2 = ax1.twinx()  
    color_ret = 'tab:red'
    ax2.set_ylabel('Ret Pred Level MSP500', color=color_ret)
    ax2.plot(df['eom'], df['Pred. Ret'], color=color_ret, label='ret_pred', marker='d')
    ax2.tick_params(axis='y', labelcolor=color_ret)
    
    plt.title('TR vs Ret Pred Level MSP500 (Dual Axis)')
    fig.tight_layout()
    plt.show()
#%% Portfolio Analysis

# ================================
# Predictability by DolVol
# ================================
"""
High DolVol stocks do not, in general, have higher predictability.
"""

estimators = [
          'XGBRegHPlenient_LevelTrMsp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          'TransformerSet_Dropout010_LevelTrMSp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          'RFF_LevelTrMsp500Target_SP500UniverseFL_ZscoreFeatures_RollingWindow_win120_val12_test12',
          'IPCA_LevelTrMsp500Target_CRSPUniverse_ZscoreFeatures_RollingWindow_win120_val12_test12',
          ]

#Load return predictions
#At 'eom', predictions are for eom+1
df_retPred = GF.load_MLpredictions(Models, estimators) 

prediction_cols = list(df_retPred.columns.drop(['id','eom']))

sp500_constituents = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      ).assign(in_sp500 = True)

df_dolvol = pd.read_sql_query(("SELECT id, eom, dolvol_126d FROM Factors_processed "
                               f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                               f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                               ),
                              con = JKP_Factors,
                              parse_dates = {'eom'})

df_retPred = (df_retPred
              .merge(sp500_constituents, on = ['id','eom'], how = 'left')
              .pipe(lambda df: df.loc[df['in_sp500'] == True])
              .drop('in_sp500', axis = 1)
              .pipe(lambda df: df[(df['eom'] >= trading_start - pd.offsets.MonthEnd(1)) 
                                  & 
                                  (df['eom'] <= trading_end)]
                    )
              .pipe(lambda df: df.merge(df_returns[['id','eom','tr_ld1']], on = ['id','eom'], how = 'left'))
              .pipe(lambda df: df.merge(df_dolvol, how = 'left', on = ['id','eom']))
              .sort_values(by = ['eom','id'])
              .reset_index(drop = True)
              )

df_retPred['quintile'] = (
    df_retPred.groupby('eom')['dolvol_126d']
    .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
)

dummies = pd.get_dummies(df_retPred['quintile'] + 1, prefix='dolvol')

# 3. Join them back to the original dataframe
df_retPred = pd.concat([df_retPred, dummies], axis=1).drop('quintile', axis=1)

results = {}
for predictor in prediction_cols:
    
    results[predictor] = {}

    for quintile in list(dummies.columns):
                
        data = df_retPred.loc[df_retPred[quintile] == 1][['id','eom',predictor, 'tr_ld1']]
        
        data = data.set_index(['id', 'eom'])
        
        mod = PanelOLS(data['tr_ld1'], data[predictor], time_effects=True)
        
        res = mod.fit(cov_type='clustered', cluster_time=True)

        # Store the coefficient and p-value in the sub-dictionary
        results[predictor][quintile] = {
            'slope': res.params[predictor],
            'p_value': res.pvalues[predictor]
        }
        
#%% Portfolio: Large Cap Stocks

"""
Compare 
ratio of value-weighted MarketCap of strategy
                vs. 
value-weighted MarketCap of S&P500 

Higher tc strats have higher market cap than index
"""

df_me = pd.read_sql_query("SELECT id, eom, me From Factors_processed",
                          con = JKP_Factors,
                          parse_dates = {'eom'})

df_me['sp500_weight'] = df_me.groupby('eom')['me'].transform(lambda x: x/x.sum())

df_sp500_me = df_me.assign(me_sp500 = lambda df: df['me'] * df['sp500_weight'])
df_sp500_me = df_sp500_me.groupby('eom')['me_sp500'].sum().reset_index()


tc_values       = [1.0, 0.5, 0.1, 0.01]
tc_map          = {1.0: "High", 0.5: "Med", 0.1: "Low", 0.01: "Tiny"}
flatMaxPiVal    = 0.15
volScaler       = 1.0

# ---- Predictors ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Labels ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# Master container: {Model_Label: {TC_Status: {Metrics}}}
all_results = {label: {} for label in labels}

# Loop over Strats
for tc_val in tc_values:
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = flatMaxPiVal,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tc_val, 
                        )

    # ---- Load Strats ----
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        if Ridge:
            load_string = "Ridge_" + load_string
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f), labels[i]])
    
    for item in strats:
        df_strat    = item[0]['Strategy']
        label       = item[1]
            

        df_strat['eom_lag'] = df_strat['eom'] - pd.offsets.MonthEnd(1)


        df_strat = df_strat.merge(df_me[['id','eom', 'me', 'sp500_weight']], how = 'left', 
                          left_on = ['id','eom_lag'],
                          right_on = ['id','eom'], 
                          suffixes = ('','_y')).drop('eom_y', axis=1)
        
        df_strat = df_strat.assign(me_w = lambda df: df['me']*df['pi'])
        
        df_sum = df_strat.groupby('eom_lag')['me_w'].sum().reset_index()
        
        df_sum = df_sum.merge(df_sp500_me, left_on = ['eom_lag'],
                              right_on = ['eom'], 
                              how = 'left')
        
        df_sum = df_sum.assign(ratio = lambda df: df['me_w'] / df['me_sp500'])

#%% Fama-French Regressions

# ---- Load Portfolios ----
portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 1,       # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        )
strats = ["XGB", "Transformer", "IPCA", "RFF"]
tc_scale = 0.01 

# ---- Read in FF5 ----
FF_market = True # Results are basically the same
df_FF5 = (pd.read_csv(path + "Data/FF5.csv")
          .assign(eom = lambda df: pd.to_datetime(df['dateff']))
          .drop('dateff', axis = 1)
          )

# ---- Read in Pastor-Stambaugh Liquidity Factor ----
# Level:    Level of Aggregate Liquidity (non-tradeable)
# Innov:    Innovations in Aggregate Liquidity (non-tradeable)
# VWF:      Traded Liquidity Factor (tradeable & value-weighted)
# -99:      Indicates that value is missing
df_PS = (pd.read_csv(path + "Data/" + "Pastor_Stambaugh_LiquidityFactor.csv")
         .assign(eom = lambda df: pd.to_datetime(df['DATE']))
         .drop(['DATE', 'PS_LEVEL', 'PS_INNOV'], axis = 1)
         .pipe(lambda df: df.loc[df['PS_VWF'] > -99])
         )

# ---- Read in own Factors ----
df_ff_own = build_mega_liq_factors(sp500_only = True, q1 = [0.8], q2 = [0.8])

# Merge Factors together and make date last day of the month instead of last trading day
df_factors = (df_FF5.merge(df_PS, on = ['eom'], how = 'left')
              .assign(eom = lambda df: df['eom'] + pd.offsets.MonthEnd(0))
              .merge(df_ff_own, on = ['eom'], how = 'left')
              )


# ---- Build additional Momentum Factors ----
# Caution: Portfolios are long-only as otherwise TC cannot be computed in the same fashion as done here
for name in ["ret_1_0", "ret_3_1", "ret_6_1", "ret_9_1", "ret_12_1", "ret_12_7", "ret_60_12"]:
    df_strat, df_profit = long_short_portfolio(df_mom, name,  # Predicted Returns
                             df_returns, # Realised Returns
                             df_me, # Market Equity for Value Weighting
                             long_cutoff = 0.9,  
                             short_cutoff = 0.1,
                             value_weighted = True,
                             long_only = False,
                             )
    df_profit = df_profit.rename(columns = {'cumret':f"cumret_gross_{name}_LS_Decile",
                                          'ret':f"ret_gross_{name}_LS_Decile"})
    
    # Merge to existing factors
    df_factors = df_factors.merge(df_profit[['eom', f"ret_gross_{name}_LS_Decile"]],
                                  on = ['eom'], how = 'left')


# ---- Check which factor actually contains new information (risk) ----
df_factors.corr()

# ---- Select subset of factors ----

# Define different Factor Regressions
if FF_market: 
    cols_FF3 = ['mktrf', 'smb', 'hml']
else:
    df_factors = (df_factors.merge(df_spy[['eom','ret']], on = ['eom'], how = 'left')
                  .assign(ret = lambda df: df['ret'] - df['rf'])
                  .rename(columns = {'ret': 'SP500mRF'})
                  )
    cols_FF3 = ['SP500mRF', 'smb', 'hml']

cols_FF5 = cols_FF3 + ['rmw', 'cma']
cols_FF5Mom = cols_FF5 + ['umd']
cols_FF5MomLiq = cols_FF5Mom + ['PS_VWF']
cols_self = cols_FF5MomLiq + ['ret_gross_ret_1_0_LS_Decile', 'ret_gross_ret_3_1_LS_Decile', 
                              #'ret_gross_ret_6_1_LS_Decile', 
                              #'ret_gross_ret_9_1_LS_Decile', 'ret_gross_ret_12_1_LS_Decile',
                              #'ret_gross_ret_12_7_LS_Decile', 
                              'ret_gross_ret_60_12_LS_Decile'] 
#uMD is highly correlated many other momentum factors (>0.7, so they don't add meaningful information)
#The included momentum factors are also highly correlated with the omitted ones, so even more reasons to omit the omitted ones

"""
# Long Momentum Factors
df_mom_fact = pd.read_sql_query("SELECT * from Other_Portfolios",
                                con = db_OtherPortfolios, 
                                parse_dates = {'eom'})
mask = ((df_mom_fact.columns.str.startswith('ret_net_0.01')) 
        & 
        (~df_mom_fact.columns.str.contains('OneOverN'))
        | 
        (df_mom_fact.columns == 'eom')
        )
df_mom_fact = df_mom_fact.loc[:, mask]
df_factors = df_factors.merge(df_mom_fact, on = ['eom'], how = 'left')

# As momentum factors long, must deduct market factor or else the correlation with the market factor is > 0.85
for col in df_mom_fact.drop('eom',axis = 1).columns:
    df_factors[col] = df_factors[col] - df_factors['mktrf'] 
"""

# Return column
ret_col = f"ret_net_{tc_scale}" 

# list of factors used
regime = cols_self 

# Container for results
results = {strat: {} for strat in strats}

for strat in strats:
    
    # Extract Portfolio Return
    df_profit = portfolios["hypo"][strat]['Profit']
        
    # Combine everything into one dataframe
    data = (df_profit[['eom', ret_col]].merge(df_factors, on = ['eom'], how = 'left')
            # Portfolio return must be in excess of risk-free rate!
            .assign(retMrf = lambda df: df[ret_col] - df['rf'])
            .merge(df_spy[['eom','ret']].rename(columns = {'ret':'ret_sp500'}), on = 'eom', how = 'left')
            .assign(retMsp500 = lambda df: df[ret_col] - df['ret_sp500'])
            ).dropna()
    
    X = data[regime].assign(constant = 1)
    y = data['retMsp500'] # yields same result data[ret_col] - data['ret_sp500']
    
    model = sm.OLS(y, X)

    reg = model.fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 6}
    )
    print(f"===== {strat} =====")
    print(reg.summary())
    results[strat] = reg
    

reg_coef_names = {'mktrf': 'Mkt', 'smb':'SMB', 'hml': 'HML', 'rmw': 'RMW', 'cma': 'CMA', 'umd': 'UMD',
             'PS_VWF': 'PS Liq', 'MegaLiq': 'DolVol', 'MegaCap': 'MegaCap', 'constant':r"$\alpha_{ann}$",
             'ret_gross_ret_1_0_LS_Decile': "Mom 1-0", 'ret_gross_ret_3_1_LS_Decile': "Mom 3-1",
             'ret_gross_ret_60_12_LS_Decile': 'Mom 60-12', 'SP500mRF': 'Mkt'}

# Build table body
header = " & " + " & ".join(strats) + r" \\"
lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\caption{Regression: Portfolio Returns on Factors.}")
lines.append(r"\label{Table:Reg_PF_on_FF}")
lines.append(r"\begin{tabular}{l" + "c"*len(strats) + r"}")
lines.append(r"\toprule")
lines.append(header)
lines.append(r"\midrule")

# Fill each row with the coefficient value
rows = []
for reg_coef in reg.params.index:
    reg_coef_name = reg_coef_names[reg_coef]
    row_entry = f"{reg_coef_name} "
    for strat in strats:
        regression = results[strat]
        coef = float(regression.params.loc[reg_coef])
        pval = float(regression.pvalues.loc[reg_coef])
        # Annualise alpha
        if reg_coef_name == r"$\alpha_{ann}$":
            coef = coef*12
            
        row_entry += f" & {coef:.3f}{star_notation(pval)}"
    row_entry += r" \\"
    rows.append(row_entry)
lines += rows
lines += [r"\midrule"]

# R2
r2_vals  = r"$R^2$"

for strat in strats:
    r2_vals += f" & {results[strat].rsquared:.3f}"
r2_vals += r" \\"
lines += [r2_vals]

# Table end
lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

latex_table = "\n".join(lines)
print(latex_table)

#%% STMOM

db_crsp = sqlite3.connect(path + "Data/CRSP_monthly.db")

df_crsp = pd.read_sql_query(("SELECT PERMNO as id, eom, PrimaryExch, MthCap, MthRet, MthVol, (ShrOut * 1000) AS ShrOut "
                             "FROM CRSP_monthly "
                             f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                             f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"),
                            con = db_crsp,
                            parse_dates = {'eom'})

# Get leaded Return
df_crsp = (df_crsp
           .merge(df_crsp.assign(eom_lag = lambda df: df['eom'] - pd.offsets.MonthEnd(1))
                  .rename(columns = {'MthRet':'tr_ld1'})
                  [['id','eom_lag', 'tr_ld1']],
                  left_on = ['id','eom'], 
                  right_on = ['id','eom_lag'],
                  how = 'left')
           ).drop('eom_lag', axis = 1)

df_crsp = df_crsp.loc[(df_crsp['MthVol'] > 0) & (df_crsp['ShrOut'] > 0)].dropna()
df_crsp = df_crsp.assign(TO = lambda df: df['MthVol']/df['ShrOut'])



def build_ms_stmom_factor(df_crsp, q_ret=None, q_to=None):
    """
    Build Medhat-Schmeling (2021) style STMOM and STREV factors using
    conditional double sorts with NYSE ('N') breakpoints.

    Inputs (df_crsp must have):
      - eom (month end, datetime)
      - id
      - PrimaryExch (NYSE flag 'N')
      - MthCap (market cap for value weights)
      - tr_ld1 (next-month return)
      - MthRet (formation return, prior month)
      - TO (share turnover, prior month)

    Returns:
      - factor_df with columns ['eom', 'STMOM', 'STREV']
      - (optional) port_ret table if you want to inspect 10x10 portfolios
    """

    # ------------------------------------------------------------------
    # Defaults: deciles
    # ------------------------------------------------------------------
    if q_ret is None:
        q_ret = [i/10 for i in range(1, 10)]   # 0.1,...,0.9
    if q_to is None:
        q_to = [i/10 for i in range(1, 10)]    # 0.1,...,0.9

    labels_ret = list(range(1, 11))  # 1..10 losers..winners
    labels_to  = list(range(1, 11))  # 1..10 low..high turnover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wavg(x, ret_col, w_col):
        r = x[ret_col]
        w = x[w_col]
        mask = r.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return np.average(r[mask], weights=w[mask])

    def assign_portfolio(df, sorting_variable, q, labels, bp_df=None):
        """
        Assign portfolios using quantile breakpoints computed from bp_df
        (NYSE-only), applied to df (full universe).

        Keeps it simple: uses Series.quantile on bp_df and pd.cut on df.
        """
        if bp_df is None:
            bp_df = df

        x_bp = bp_df[sorting_variable].dropna()
        x    = df[sorting_variable]

        if x_bp.empty:
            return pd.Series(np.nan, index=df.index)

        probs = [0.0] + list(q) + [1.0]
        cuts = x_bp.quantile(probs).to_numpy()

        # Ensure strictly increasing cutpoints (avoid pd.cut errors)
        cuts = np.unique(cuts)
        if len(cuts) < 2:
            return pd.Series(np.nan, index=df.index)

        # If duplicates dropped, we must also drop labels accordingly
        # Number of bins = len(cuts)-1
        n_bins = len(cuts) - 1
        use_labels = labels[:n_bins]

        return pd.cut(
            x,
            bins=cuts,
            labels=use_labels,
            include_lowest=True,
            right=True
        )

    # ------------------------------------------------------------------
    # Core: conditional double sort with NYSE breakpoints
    # ------------------------------------------------------------------
    df = df_crsp.copy()

    # Require formation variables and next-month return for realized performance
    df = df[df["MthRet"].notna() & df["TO"].notna() & df["tr_ld1"].notna() & df["MthCap"].notna()]
    df = df[df["MthCap"] > 0]

    # --- Outer sort on prior-month return using NYSE breakpoints ---
    df["PF_Ret"] = (
        df.groupby("eom", group_keys=False)
          .apply(lambda g: assign_portfolio(
              g,
              sorting_variable="MthRet",
              q=q_ret,
              labels=labels_ret,
              bp_df=g[g["PrimaryExch"] == "N"]   # NYSE-only breakpoints
          ))
    )

    # --- Inner sort on turnover within (month, return decile) using NYSE breakpoints ---
    df["PF_TO"] = (
        df.groupby(["eom", "PF_Ret"], group_keys=False)
          .apply(lambda g: assign_portfolio(
              g,
              sorting_variable="TO",
              q=q_to,
              labels=labels_to,
              bp_df=g[g["PrimaryExch"] == "N"]   # NYSE-only breakpoints within return bin
          ))
    )

    # ------------------------------------------------------------------
    # Compute 10x10 portfolio returns: value-weighted by MthCap (formation ME)
    # Returns are next-month returns tr_ld1
    # ------------------------------------------------------------------
    port_ret = (
        df.dropna(subset=["PF_Ret", "PF_TO", "tr_ld1"])
          .groupby(["eom", "PF_Ret", "PF_TO"])
          .apply(lambda x: _wavg(x, "tr_ld1", "MthCap"))
          .rename("ret")
          .reset_index()
    )

    # Pivot to easy access
    grid = (
        port_ret.set_index(["eom", "PF_Ret", "PF_TO"])["ret"]
                .unstack(["PF_Ret", "PF_TO"])
    )

    # ------------------------------------------------------------------
    # Factors:
    #   STMOM_t = (Winners - Losers) within highest turnover decile
    #           = P(R=10, TO=10) - P(R=1, TO=10)
    #
    #   STREV*_t = (Winners - Losers) within lowest turnover decile
    #            = P(R=10, TO=1) - P(R=1, TO=1)
    # ------------------------------------------------------------------
    def _get_col(R, TO):
        return (R, TO)

    factor = pd.DataFrame({"eom": grid.index}).set_index("eom")

    # Safely handle missing columns (e.g., if some month lacks enough NYSE names)
    for name, (Rw, Rl, TOd) in {
        "STMOM": (10, 1, 10),
        "STREV": (10, 1, 1)
    }.items():
        col_w = _get_col(Rw, TOd)
        col_l = _get_col(Rl, TOd)
        if col_w in grid.columns and col_l in grid.columns:
            factor[name] = grid[col_w] - grid[col_l]
        else:
            factor[name] = np.nan

    factor = factor.reset_index().sort_values("eom").reset_index(drop=True)
    return factor, port_ret

ms_factor, ms_ports = build_ms_stmom_factor(df_crsp)

#%% Other Robustness Portfolios

"""
Compute Net and Gross performance of other easily implementable long-only
portfolios on the S&P 500.
"""

# Generals
df_profit_collector = pd.DataFrame()

# ============================================================
#   One over N - Obviously performance super close to S&P 500
# ============================================================

# Load arbitrary strategy dataframe

run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 1.0, 
                    )

load_string = (f"{settings_string(run_settings)}_" 
               f"XGBRegHPlenient_" 
               f"LevelTrMsp500Target_" 
               f"SP500UniverseFL_" 
               f"RankFeatures_"
               f"RollingWindow_win120_val12_test12.pkl")
    
with open(path + f"Portfolios/{load_string}", "rb") as f:
    df_strat = (pickle.load(f)['Strategy']
                .drop(columns = ["pi_g_tm1", "pi", "rev", "tc", "lambda"])
                )
    
# One over N portfolio weight
df_strat = df_strat.assign(pi = lambda df: 1 / df.groupby('eom')['id'].transform('count'))

df_strat, df_profit = compute_TC_from_Scratch(df_strat, df_wealth, df_returns, 
                                              df_sp500_ids,
                                              tc_scalers = [1.0, 0.5, 0.1, 0.01] , 
                                              name = "OneOverN")

if df_profit_collector.empty:
    df_profit_collector = df_profit
else:
    # Merge on eom to keep everything in one wide table
    df_profit_collector = df_profit_collector.merge(df_profit, on='eom', how='outer')

# ======================
#   Momentum Portfolio
# ======================

"""
Return-based momentum and reversal characteristics.

All variables are constructed from the return index RI_t
(Return index also contains dividends as opposed to merely looking at the price)

 and follow the rule:

    ret_X_Y = (RI_{t-X} / RI_{t-Y}) - 1

That is, the cumulative return from month t-Y up to month t-X.

Variables
---------
ret_1_0
    Short-term reversal (1-month return).

    Construction:
        (RI_t / RI_{t-1}) - 1

ret_3_1
    Momentum over months 1 to 3 (skips the most recent month).

    Construction:
        (RI_{t-1} / RI_{t-3}) - 1

ret_6_1
    Momentum over months 1 to 6.

    Construction:
        (RI_{t-1} / RI_{t-6}) - 1

ret_9_1
    Momentum over months 1 to 9.

    Construction:
        (RI_{t-1} / RI_{t-9}) - 1

ret_12_1
    Momentum over months 1 to 12 (standard 12-month momentum).

    Construction:
        (RI_{t-1} / RI_{t-12}) - 1

ret_12_7
    Momentum over months 7 to 12 (skip-recent momentum).

    Construction:
        (RI_{t-7} / RI_{t-12}) - 1

ret_60_12
    Long-term momentum over months 12 to 60.

    Construction:
        (RI_{t-12} / RI_{t-60}) - 1

Notes
-----
These definitions follow the Global Factor Data documentation (Momentum/Reversal
section) and are standard in the empirical asset pricing literature. :contentReference[oaicite:0]{index=0}
"""

# Compute Net Return of long-only momentum strategies
for name in ["ret_1_0", "ret_3_1", "ret_6_1", "ret_9_1", "ret_12_1", "ret_12_7", "ret_60_12"]:

    #If Decile Portfolio desired (else Top10 Portfolio)
    decile = True
    # Caution: Portfolios are long-only as otherwise TC cannot be computed in the same fashion as done here
    
    df_strat, df_profit = long_short_portfolio(df_mom, name,  # Predicted Returns
                             df_returns, # Realised Returns
                             df_me, # Market Equity for Value Weighting
                             long_cutoff = 0.9 if decile else 0.98,  #else Top 10
                             short_cutoff = 0.1,
                             value_weighted = True,
                             long_only = True,
                             )
    
    df_gross = df_profit
    if decile:
        df_gross = df_gross.rename(columns = {'cumret':f"cumret_gross_{name}_Decile",
                                              'ret':f"ret_gross_{name}_Decile"})
    else:
        df_gross = df_gross.rename(columns = {'cumret':f"cumret_gross_{name}_Top10",
                                              'ret':f"ret_gross_{name}_Top10"})
    
    df_strat = (df_strat
                .drop(columns = ['position','me','ret'])
                .rename(columns = {'weight':'pi'})
                )
    
    # Recompute TC as long_short_portfolio() does not compute TC. 
    # CAUTION: Transaction Costs can only be computed for long-only portfolios!
    df_strat, df_profit = compute_TC_from_Scratch(df_strat, df_wealth, df_returns, 
                                                  df_sp500_ids,
                                                  tc_scalers = [1.0, 0.5, 0.1, 0.01] , 
                                                  name = name + "_Decile" if decile else name + "_Top10")
    
    df_profit = df_profit.merge(df_gross, on = ['eom'], how = 'left')
    
    if df_profit_collector.empty:
        df_profit_collector = df_profit.copy()
    else:
        # Merge on eom to keep everything in one wide table
        df_profit_collector = df_profit_collector.merge(df_profit, on='eom', how='outer')
        

# ========================================
# Long Decile Based on return predictions
# ========================================

# ---- Predictor ----
predictors = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF',
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              ]

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Load Return Predictions ----
predictions = []
for i, predictor in enumerate(predictors):
    # Recall that at date 'eom' the prediction is for 'eom' + 1 
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    
    # If operating on SP500 Universe only
    df = df.merge(df_sp500_ids.assign(in_sp500 = True), on = ['id', 'eom'], how = 'left')
    df = df.dropna().drop(columns = 'in_sp500')
    
    predictions.append([df, labels[i]])
    
# ---- Compute Long Portfolios ----
for df_pred, label in predictions:
    # 1. Identify the prediction column dynamically
    # It's the column that isn't 'id' or 'eom'
    pred_col = [col for col in df_pred.columns if col not in ['id', 'eom']][0]
    
    print(f"Processing {label} using column: {pred_col}")
    
    # Long-Short Portfolio
    df_strat, df_profit = long_short_portfolio(
        df_pred,
        prediction_col=pred_col,
        df_returns=df_returns, 
        df_me=df_me,
        long_cutoff=0.9, 
        short_cutoff=0.1,
        value_weighted=True, 
        long_only = True
    )
    
    df_gross = df_profit
    df_gross = df_gross.rename(columns = {'cumret':f"cumret_gross_LongDecile_{label}",
                                          'ret':f"ret_gross_LongDecile_{label}"})
    
    df_strat = (df_strat
                .drop(columns = ['position','me','ret'])
                .rename(columns = {'weight':'pi'})
                )
    
    df_strat, df_profit = compute_TC_from_Scratch(df_strat, df_wealth, df_returns, 
                                                  df_sp500_ids,
                                                  tc_scalers = [1.0, 0.5, 0.1, 0.01] , 
                                                  name = f"_LongDecile_{label}")
    
    df_profit = df_profit.merge(df_gross, on = ['eom'], how = 'left')

    
    if df_profit_collector.empty:
        df_profit_collector = df_profit.copy()
    else:
        # Merge on eom to keep everything in one wide table
        df_profit_collector = df_profit_collector.merge(df_profit, on='eom', how='outer')


# ===========================================
# Fama-French Test portfolios on the S&P 500
# ===========================================





# =============================================
#  Final Output: Table of Risk-adjusted Returns
# =============================================

cols_lowest_tc = df_profit_collector.filter(regex='^ret_net_0.01')

results = []
for col in cols_lowest_tc:
    # 1. Create a temporary DF with 'eom' and the specific strategy return
    # This is necessary because your functions use .merge(on='eom')
    df_temp = df_profit_collector[['eom', col]].copy()
    
    # 2. Compute Metrics
    mu_ann, sigma_ann = meanRet_varRet(df_temp, col)
    sharpe = SharpeRatio(df_temp, risk_free, col)
    ir = InformationRatio(df_temp, df_spy, col, 'ret', risk_free)
    
    # 3. Append as a dictionary
    results.append({
        'Strategy': col,
        'mu': mu_ann,
        'sigma': sigma_ann,
        'Sharpe': sharpe,
        'Information_Ratio': ir
    })
    

cols_gross = df_profit_collector.filter(regex='^ret_gross')
for col in cols_gross:
    # 1. Create a temporary DF with 'eom' and the specific strategy return
    # This is necessary because your functions use .merge(on='eom')
    df_temp = df_profit_collector[['eom', col]].copy()
    
    # 2. Compute Metrics
    mu_ann, sigma_ann = meanRet_varRet(df_temp, col)
    sharpe = SharpeRatio(df_temp, risk_free, col)
    ir = InformationRatio(df_temp, df_spy, col, 'ret', risk_free)
    
    # 3. Append as a dictionary
    results.append({
        'Strategy': col,
        'mu': mu_ann,
        'sigma': sigma_ann,
        'Sharpe': sharpe,
        'Information_Ratio': ir
    })
    
df_metrics = pd.DataFrame(results).set_index('Strategy')

name_mapping = {
    'ret_net_0.01_OneOverN': '1/N Portfolio',
    'ret_net_0.01_ret_1_0_Decile': 'Momentum 1-0',
    'ret_net_0.01_ret_3_1_Decile': 'Momentum 3-1',
    'ret_net_0.01_ret_6_1_Decile': 'Momentum 6-1',
    'ret_net_0.01_ret_9_1_Decile': 'Momentum 9-1',
    'ret_net_0.01_ret_12_1_Decile': 'Momentum 12-1',
    'ret_net_0.01_ret_12_7_Decile': 'Momentum 12-7',
    'ret_net_0.01_ret_60_12_Decile':'Momentum 60-12',
    'ret_net_0.01__LongDecile_XGB': 'XGB',
    'ret_net_0.01__LongDecile_Transformer': 'Transformer',
    'ret_net_0.01__LongDecile_IPCA': 'IPCA',
    'ret_net_0.01__LongDecile_RFF': 'RFF'}

df_metrics = df_metrics.rename(index=name_mapping)

latex_str = df_metrics.to_latex(
    index=True, 
    column_format='lcccc', # 'l' for left index, 'c' for four centered columns
    float_format="%.3f",    # Forces 3 decimal places even for values like 1.100
    bold_rows=False,
)

# ================
# Save all results
# ================

df_profit_collector.to_sql(name = "Other_Portfolios", con = db_OtherPortfolios, if_exists = 'replace')
db_OtherPortfolios.close()



#%% Information & Sharpe Ratio Statistical Tests


# ---- Functions Information Ratio ----
def information_ratio(active):
    """IR = mean(active) / std(active), using sample std (ddof=1)."""
    active = pd.Series(active).dropna()
    return float(active.mean() / active.std(ddof=1)) * np.sqrt(12)

def ir_diff_block_bootstrap(active1, active2, B=20_000, block_size=6, seed=42):
    """
    Tests H1: IR1 - IR2 > 0 using a moving block bootstrap.
    Returns point estimate, 95% CI, and one-sided p-value for diff > 0. 

    active1, active2: pd.Series aligned by date 
    """
    
    rng = np.random.default_rng(seed)
    d = pd.concat([active1, active2], axis=1).dropna()
    x = d.iloc[:, 0].to_numpy()
    y = d.iloc[:, 1].to_numpy()
    T = len(x)

    def mbb_indices():
        k = int(np.ceil(T / block_size))
        starts = rng.integers(0, T - block_size + 1, size=k)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        return idx

    ir1 = information_ratio(x)
    ir2 = information_ratio(y)
    diff0 = ir1 - ir2

    diffs = np.empty(B, dtype=float)
    for i in range(B):
        idx = mbb_indices()
        diffs[i] = information_ratio(x[idx]) - information_ratio(y[idx])

    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    p_one_sided = float(np.mean(diffs <= 0.0))  # for H1: diff > 0

    return {
        "n_obs": int(T),
        "IR1": float(ir1),
        "IR2": float(ir2),
        "diff": float(diff0),
        "ci_95": (float(ci_lo), float(ci_hi)),
        "p_value_greater": p_one_sided,
        "B": int(B),
        "block_size": int(block_size),
    }
# -------------------------------------------------

# ---- Functions Sharpe Ratio ----
def sr(ret_exc):
    """IR = mean(active) / std(active), using sample std (ddof=1)."""
    ret_exc = pd.Series(ret_exc).dropna()
    return float(ret_exc.mean() / ret_exc.std(ddof=1)) * np.sqrt(12)

def sr_diff_block_bootstrap(ret_exc1, ret_exc2, B=20_000, block_size=6, seed=42):
    """
    Tests H1: IR1 - IR2 > 0 using a moving block bootstrap.
    Returns point estimate, 95% CI, and one-sided p-value for diff > 0.

    ret_exc1, ret_exc2: pd.Series aligned by date (will be aligned internally).
    """
    rng = np.random.default_rng(seed)
    d = pd.concat([ret_exc1, ret_exc2], axis=1).dropna()
    x = d.iloc[:, 0].to_numpy()
    y = d.iloc[:, 1].to_numpy()
    T = len(x)

    def mbb_indices():
        k = int(np.ceil(T / block_size))
        starts = rng.integers(0, T - block_size + 1, size=k)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        return idx

    ir1 = information_ratio(x)
    ir2 = information_ratio(y)
    diff0 = ir1 - ir2

    diffs = np.empty(B, dtype=float)
    for i in range(B):
        idx = mbb_indices()
        diffs[i] = information_ratio(x[idx]) - information_ratio(y[idx])

    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    p_one_sided = float(np.mean(diffs <= 0.0))  # for H1: diff > 0

    return {
        "n_obs": int(T),
        "IR1": float(ir1),
        "IR2": float(ir2),
        "diff": float(diff0),
        "ci_95": (float(ci_lo), float(ci_hi)),
        "p_value_greater": p_one_sided,
        "B": int(B),
        "block_size": int(block_size),
    }
# -------------------------------------------------

# Load Portfolios
portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 1,       # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        )

# Strats
strats = ["XGB", "Transformer", "IPCA", "RFF"]

tc_map = {1.0: "Large", 0.5: "Med", 0.1: "Small", 0.01: "Tiny"}

combined_results = []

for strat in strats:
    for tc_scaler in [0.5, 0.1, 0.01]:
    
        # Load Portfolios
        ret_col_actual = "ret_net"
        df_profit_actual   = portfolios["actual"][strat][tc_scaler]['Profit']
        #df_strategy_actual = portfolios["actual"][strat][tc_scaler]['Strategy']
    
        # Get column name
        ret_col_hypo = f"ret_net_{tc_scaler}"
        # Extract Profit of strat and tc combo
        df_profit_hypo   = portfolios["hypo"][strat]['Profit']
        #df_strategy_hypo = portfolios["hypo"][strat]['Strategy']
        
        # ---- Testing the Information Ratio ----
        # Get Strategy Return
        ret_hypo   = df_profit_hypo[ret_col_hypo]
        ret_actual = df_profit_actual[ret_col_actual]
        
        # Get S&P 500 Return
        sp500_ret = (df_spy
                     # Same dates as strategies
                     .loc[df_spy['eom'].isin(df_profit_hypo.eom.unique())]
                     # sort dates (ascending)
                     .sort_values(by = "eom")
                     ['ret']
                     )
         
        ir_test = ir_diff_block_bootstrap(ret_hypo-sp500_ret,   # IR Act as Large
                                      ret_actual-sp500_ret)     # IR Act as Pay as
        
        print("IR", strat , tc_scaler, ir_test["p_value_greater"])
        
        # ---- Testing the Sharpe Ratio ----
        retexc_hypo = df_profit_hypo.merge(risk_free, on = 'eom', how = 'left')
        retexc_hypo['ret_exc'] = retexc_hypo[ret_col_hypo] - retexc_hypo['rf']
        
        retexc_actual = df_profit_actual.merge(risk_free, on = 'eom', how = 'left')
        retexc_actual['ret_exc'] = retexc_actual[ret_col_actual]- retexc_actual['rf']


        sr_test = sr_diff_block_bootstrap(retexc_hypo['ret_exc'], retexc_actual['ret_exc'])
        print("SR", strat, tc_scaler, sr_test['p_value_greater'])
        
        combined_results.append({
            "Strat": strat,
            "PayAs": tc_map[tc_scaler],
            # Sharpe Stats
            "SR_VoT": sr_test["IR1"], 
            "SR_Pay": sr_test["IR2"],
            "SR_Delta": sr_test["diff"],
            "SR_Pval": sr_test["p_value_greater"],
            # IR Stats
            "IR_VoT": ir_test["IR1"],
            "IR_Pay": ir_test["IR2"],
            "IR_Delta": ir_test["diff"],
            "IR_Pval": ir_test["p_value_greater"]
        })

# Build LaTeX String
# Define the mapping for alignment
phantom_map = {
    "***": "***",
    "**":  "**\\phantom{*}",
    "*":   "*\\phantom{**}",
    "":    "\\phantom{***}"
}

latex_str = r"""\begin{table}[H]
    \centering
    \caption{Performance Comparison: Sharpe and Information Ratios}
    \begin{tabular}{lcccccc}
        \toprule
        Pay as & SR (VoT) & SR & $\Delta$ SR & IR (VoT) & IR & $\Delta$ IR \\
        \midrule"""
        
for strat in strats:
    # Add Section Header
    strat_display = strat
    latex_str += f"\n        \multicolumn{{7}}{{c}}{{\\textbf{{{strat_display}}}}} \\\\"
    
    # Filter data for this strategy
    strat_rows = [r for r in combined_results if r['Strat'] == strat]
    
    for row in strat_rows:
        # Get stars for both deltas
        sr_stars = star_notation(row['SR_Pval'])
        ir_stars = star_notation(row['IR_Pval'])
        
        sr_stars = phantom_map.get(sr_stars, "\\phantom{***}")
        ir_stars = phantom_map.get(ir_stars, "\\phantom{***}")
        
        # Formatting rows
        latex_str += (
            f"\n        {row['PayAs']} & "
            f"{row['SR_VoT']:.3f} & {row['SR_Pay']:.3f} & {row['SR_Delta']:.3f}{sr_stars} & "
            f"{row['IR_VoT']:.3f} & {row['IR_Pay']:.3f} & {row['IR_Delta']:.3f}{ir_stars} \\\\"
        )
    latex_str += "\n        \midrule"

# Finalize formatting
latex_str = latex_str.rsplit('\\midrule', 1)[0] + r"""\bottomrule
    \end{tabular}
\end{table}"""

print(latex_str)

#%% Permuted Portfolios

# ============
# Load Data
# ===========
strats = ["XGB", "Transformer", "IPCA", "RFF"]

portfolios_permute = {strat: {} for strat in strats}

for strat in strats:
    with open(f"{path}Portfolios/{strat}_Permute_volScaler=1_tcScaler=1.pkl", "rb") as f:
        data = pickle.load(f)
    portfolios_permute[strat]['Strategy'] = data['Strategy'].drop('lambda', axis=1)
                           
                           
dict_dfs = {}
dict_profits = {}

portfolios = {
    "permute": {
        strat: {
            "Profit": {},
            "Strategy": {},
        }
        for strat in strats
    }}

# Scale factors for transaction costs
tc_scalers = [1.0, 0.5, 0.1, 0.01]

for strat in strats:
    df = portfolios_permute[strat]['Strategy']
    # 2. Merge auxiliary data
    # Baseline Wealth at BEGINNING of month
    df = df.merge(df_wealth.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                  [['eom_lead','wealth']],
                  left_on = ['eom'], right_on=['eom_lead'], how='left').drop(columns='eom_lead')
    # Baseline Lambda at BEGINNING of month
    df = df.merge(df_kl.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                  [['id','eom_lead','lambda']], left_on=['id', 'eom'],
                  right_on = ['id','eom_lead'], 
                  how = 'left').drop(columns='eom_lead')
    
    
    # 3. Compute Hypothetical Transaction Costs (Loop through scalers)
    for tc_scaler in tc_scalers:
        tc_col = f"tc_{tc_scaler}"
        ret_net_col = f"ret_net_{tc_scaler}"
        
        df = df.assign(**{
            tc_col: lambda x, s=tc_scaler: s * x['wealth'] * 0.5 * x['lambda'] * (x['pi'] - x['pi_g_tm1'])**2, # Caution: 0.5 due to JKMP22 legacy code.
            ret_net_col: lambda x, tc=tc_col: x['rev'] - x[tc]
        })
    
    # Store the granular dataframe
    dict_dfs[strat] = df
    
    # 4. Aggregate results for df_profit
    df_profit = None
    for tc_scaler in tc_scalers:
        ret_net_col = f"ret_net_{tc_scaler}"
        cum_ret_col = f"cum_net_{tc_scaler}"
        
        df_add = (df
                  .groupby('eom', as_index=False)
                  .agg(**{ret_net_col: (ret_net_col, 'sum')})
                  .sort_values('eom')
                  .assign(**{cum_ret_col: lambda x: (1.0 + x[ret_net_col]).cumprod()}))
    
        # --- Logic to handle missing initial df_profit ---
        if df_profit is None:
            df_profit = df_add.copy()
        else:
            # For subsequent scalers, merge into the base
            df_profit = df_profit.merge(
                df_add[['eom', ret_net_col, cum_ret_col]], 
                on='eom', 
                how='left'
            )
        
    dict_profits[strat] = df_profit
    
    
for strat in strats:
    portfolios['permute'][strat]['Strategy'] = dict_dfs[strat]
    portfolios['permute'][strat]['Profit'] = dict_profits[strat]
    
    
# ==========================
# Table Performance Metrics
# =========================
    
# Strats to display
strats = ["XGB", "Transformer", "IPCA", "RFF"]

# Naming in Table
tc_map = {1.0: "Large", 0.5: "Med", 0.1: "Small", 0.01: "Tiny"}

# Master container storing results
all_results = {strat: {} for strat in portfolios["permute"].keys()}

# Calculate Benchmark (only needs to run once)
mu_bench, sigma_bench, = meanRet_varRet(df_spy, 'ret')
Sharpe_bench = SharpeRatio(df_spy, risk_free, return_col='ret')
drawdown_bench, _, _ = MaxDrawdown(df_spy, df_spy, 'ret', 'ret')

#---- Compute Table of Performance Metrics for Portfolios ---
# Loop over Strats
for tc_scaler in [1.0, 0.5, 0.1, 0.01]:
    
    perform_dict = {}
    for strat in strats:
        # Load Portfolios
        ret_col_strat = f"ret_net_{tc_scaler}"
        # Extract Profit of strat and tc combo
        df_profit   = portfolios["permute"][strat]['Profit']
        df_strategy = portfolios["permute"][strat]['Strategy']
        
        
        mu_s, sigma_s = meanRet_varRet(df_profit, ret_col_strat)
        sharpe_s = SharpeRatio(df_profit, risk_free, return_col=ret_col_strat)
        ir = InformationRatio(df_profit, df_spy, ret_col_strat, 'ret', risk_free)
        to, _ = Turnover(df_strategy, df_wealth)
        dd, _, _ = MaxDrawdown(df_profit, df_spy, ret_col_strat, 'ret')
        down_cap, _, _, _ = CaptureRatio(df_profit, df_spy, ret_col_strat, 'ret')
        
        alpha_ann, p_value = capm_alpha(df_profit, df_spy, ret_col_strat, 'ret', risk_free)

        # Store in master dictionary
        all_results[strat][tc_map[tc_scaler]] = {
            'mu': mu_s, 'sigma': sigma_s, 'sharpe': sharpe_s,
            'turnover': to, 'ir': ir, 'dd': dd, 'down_cap': down_cap,
            'alpha': alpha_ann, 
            'alpha_p': p_value
        }

# ==========================================
# 3. Manual LaTeX Table Construction
# ==========================================
status_str = "Net" 

latex_str = f"""\\begin{{table}}[htpb]
\\centering
\\caption{{Summary Statistics {status_str} Returns}}
\\label{{Table:SummaryStats_{status_str}Returns}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{llcccccccc}}
\\toprule
Act as & Pay as & $\\mu$ & $\\sigma$ & SR & TO & IR & MaxD & DCap & $\\alpha$ \\\\
\\midrule
"""

latex_str += (
    f"$\\bullet$ & $\\bullet$ & {mu_bench:.3f} & {sigma_bench:.3f} & {Sharpe_bench:.3f} & "
    f"$\\bullet$ & 0.000 & {drawdown_bench:.3f} & 1.000 & 0.000 \\\\\n"
)
latex_str += "\\bottomrule\n"

# Model Blocks
for strat in strats:
    latex_str += "%" + "="*50 + "\n"
    latex_str += "\\toprule\n"
    latex_str += f"\\multicolumn{{10}}{{c}}{{\\textbf{{{strat}}}}} \\\\\n"
    
    # Iterate through TC statuses in specific order
    for tc_status in ["Large", "Med", "Small", "Tiny"]:
        if tc_status in all_results[strat]:
            res = all_results[strat][tc_status]
    
            # Display strat: High should be Large
            if Act_as_equalTo_Pay_as:
                act_as = tc_status 
            else:
                act_as = "Large"
    
            # Alpha with stars
            alpha_str = f"{res['alpha']:.3f}{star_notation(res['alpha_p'])}"
    
            row = (
                f"{act_as} & {tc_status} & {res['mu']:.3f} & {res['sigma']:.3f} & "
                f"{res['sharpe']:.3f} & {res['turnover']:.3f} & {res['ir']:.3f} & "
                f"{res['dd']:.3f} & {res['down_cap']:.3f} & {alpha_str} \\\\\n"
            )
            latex_str += row

    latex_str += "\\bottomrule\n"

latex_str += r"""\end{tabular}
\end{threeparttable}
\end{table}"""

print(latex_str)


# =============================
# Pairwise Comparison of alpha
# =============================
from scipy import stats

permute_rets = {}
for strat in strats:
    permute_rets[strat] = portfolios["permute"][strat]['Profit']['ret_net_0.01']
    
hypo_rets = {}
portfolios = get_strats(path = path, df_wealth = df_wealth, df_kl = df_kl,
                        flatMaxPiVal = 1,       # pi_max 
                        volScaler = 1.0,        # Volatility Benchmarking
                        )

latex_str = f"""\\begin{{table}}[htpb]
\\centering
\\caption{{Test on Excess Returns. VoT vs.\ random permute}}
\\label{{Table:Permute}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lcc}}
\\toprule
Model & $\\Delta$ & p-value \\\\
\\midrule
"""

for strat in strats:
    hypo_rets[strat] = portfolios["hypo"][strat]['Profit']['ret_net_0.01']
    
for strat in strats:
    excess_return = hypo_rets[strat] - permute_rets[strat]
    
    # 2. Perform a 1-sample t-test
    # H0: Mean of excess returns = 0
    # Ha: Mean of excess returns > 0
    t_stat, p_value_two_sided = stats.ttest_1samp(excess_return, 0)
    
    # 3. Convert to a one-sided p-value
    # Since we are testing if Mean > 0, we check the sign of the t-stat
    if t_stat > 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)

    latex_str += f"{strat} & {excess_return.mean()*12:.3f} & {p_value_one_sided:.4f} \\\\\n"

# Closing the table environments
latex_str += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textit{Note:} $\\Delta$ represents the mean of excess returns. p-values are for a one-sided test ($H_a: \\mu > 0$).
\\end{tablenotes}
\\end{threeparttable}
\\end{table}"""