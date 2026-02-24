# -*- coding: utf-8 -*- 
"""
Created on Sun Oct 26 14:13:11 2025

Overview
--------
This script implements the paper’s *predict-then-optimise* portfolio construction
pipeline for a benchmarked, long-only S&P 500 mandate with quadratic
price-impact transaction costs (Kyle’s lambda) and a volatility-benchmarking
risk constraint. It takes *one-month-ahead* return forecasts as exogenous inputs
(from ML models), then maps those forecasts into implementable portfolio weights
via a constrained myopic optimisation solved by gradient ascent in PyTorch. :contentReference[oaicite:0]{index=0}

Timing conventions (important)
------------------------------
The code distinguishes *begin-of-period* (known at the start of month t) and
*end-of-period* (realised at the end of month t) quantities. Begin-of-period
quantities are based on the information of month t-1 (since information is only
updated at the end of each month.)

Begin-of-period (known when trading at t-):
    - pi          : portfolio weights chosen at the beginning of month t
    - wealth      : AUM level used to scale transaction costs
    - Sigma       : Barra-style covariance estimate based on information up to t-1
    - g           : used to compute drifted portfolio weights
    - lambda      : Kyle’s lambda (price impact) used in quadratic transaction costs

End-of-period:
    - tr          : realised return over month t (used for realised revenue)

High-level execution flow
-------------------------
1) Load and assemble all data required for the backtest:
    - Active S&P 500 investable universe over time (constituents)
    - stock-level realised returns
    - Kyle’s lambda estimates
    - market equity (size) for benchmark-like weight bounds
    - AUM path (wealth evolution) and benchmark return (SPY) variance
    - Barra covariance objects (later converted to stock covariance matrices)
    - ML return predictions (one-month-ahead forecasts)

2) For each transaction cost regime (tc_scale):
    For each prediction model / column:
        - Run monthly rebalancing via optimise_portfolio(...)
        - Compute realised gross and net strategy returns
        - Save results (strategy weights + performance) to a pickle file whose
          name encodes run settings and model identity.

Core portfolio optimisation pipeline
------------------------------------
The monthly portfolio loop is implemented in optimise_portfolio(...). For each
trading date t:

A) Determine the tradable universe at t:
   - get_universe_partitions(prev_date=t-1, date=t, df_pf_weights)
     partitions the S&P 500 into:
        * stayers   : remain in S&P 500 from t-1 to t
        * leavers   : must be fully liquidated (pi_t = 0 enforced)
        * newcomers : newly added (no drifted portfolio weight)
        * active    : candidates for non-zero weights (stayers + newcomers)
        * zeros     : ids forced to pi_t = 0 (starts with leavers)

B) Build required inputs (using information at t-1):
   - Sigma = GF.create_cov(dict_barra[prev_date]) creates the stock return covariance
     matrix Σ_{t-1} used for the volatility constraint at t.
   - Return forecasts are pulled from df_retPred at prev_date (forecast for t).

C) Shrink universe for missing inputs:
   - shrink_universe(...) removes assets from active and adds them to zeros if
     any of the following are missing at prev_date:
        * Kyle’s lambda
        * entry in Σ
        * return prediction
     These assets are forced to pi_t = 0 to avoid trading without inputs.

D) Align/slice all objects to the active universe:
   - reduce_to_active(...) returns Σ_active, lambda_active, predictions_active
     sorted by id so vectors/matrices line up consistently in PyTorch.

E) Construct the per-period portfolio state DataFrame:
   - build_portfolio_dataframe(...) creates df_pf_t (one row per id in the
     union of stayers/newcomers/leavers) and merges:
        * Kyle’s lambda (needed for transaction costs accounting)
        * realised returns for date t (used later for realised revenue)
        * drifted weights pi_g_tm1 = g * pi_{t-1} (the “no-trade” baseline)
     It also handles key numerical details:
        * newcomers have pi_g_tm1 = 0
        * active stocks are initialised at pi_g_tm1 (warm start)
        * active newcomers get a small epsilon initial weight (softmax cannot output exactly 0)
        * zeros are set exactly to pi_t = 0

F) Solve the myopic constrained optimisation for active weights:
   - solve_pf_optimisation(...) is the numerical core. It sets up:
        * r : predicted returns vector (active)
        * S : covariance matrix (active)
        * L_diag : Kyle lambda vector (active) used as diagonal TC penalty
        * w : AUM at prev_date (scales transaction costs)
        * pi_g_tm1 : drifted lagged weights (active baseline)
     Decision variable:
        * pi_logits (unconstrained) -> pi via softmax
          This enforces long-only weights that sum to one.

     Objective (maximised via minimising a negative “loss”):
        revenue(pi) - tc(pi, pi_g_tm1, lambda, w)
     with tc = 0.5 * w * Σ_i lambda_i * (pi_i - pi_g_tm1_i)^2

     Constraints are imposed via ReLU penalties:
        * upper / lower bounds on pi_i:
            - either a flat max cap (flat_MaxPi, flat_MaxPi_limit), or
            - benchmark-relative bounds from market equity weights scaled by
              w_upperLimit and w_lowerLimit
        * volatility benchmarking:
            pi' S pi <= (SPY variance at prev_date) * vol_scaler

     Optimiser:
        * Adam over 500 iterations in float64 for numerical stability.

G) Write the solution back and compute realised accounting:
   - Active weights in df_pf_t are overwritten by the optimiser output.
   - Realised revenue: rev = pi * tr
   - Realised transaction costs:
        tc = (pi - pi_g_tm1)^2 * lambda * wealth / 2
     (note: tc_scaler is applied upstream by scaling df_kl['lambda']).

H) Update stored portfolio weights for the next month:
   - df_pf_weights is updated with the new pi so future pi_g_tm1 can be computed.

Outputs and saved artefacts
---------------------------
- optimise_portfolio returns:
    * df_strategy  : concatenation of monthly df_pf_t, including pi, lambda,
      pi_g_tm1, realised tr, rev, and tc
    * df_pf_weights: updated weights history used for subsequent months

- The main loop computes:
    * monthly net returns: sum(rev - tc) across assets per month
    * monthly gross returns: sum(rev) across assets per month
    * cumulative net/gross return series
  and saves a pickle per (tc_scale, model) run including settings and results.

Settings helpers
----------------
- value_to_token / settings_string / settings_to_id:
  Convert run settings into compact, file-system-safe identifiers used to name
  result files and to log which constraint/cost regime produced a given output.

Key implementation notes / “gotchas”
------------------------------------
- Predictions are aligned such that forecasts stored at eom=t-1 are used when
  trading at eom=t (hence the extra appended trading date in optimise_portfolio).
- Newcomers require a strictly positive initial weight for the softmax/logits
  parametrisation; otherwise log(0) and vanishing gradients can prevent entry.
- All optimisation runs use float64 (torch.float64), which is important because
  Kyle’s lambda values can be extremely small and single precision may wash out
  gradients and TC differences.
- Assets with missing inputs are forced into the zeros set to guarantee
  well-defined objective/constraints and to avoid trading on incomplete data.

Where to start reading
----------------------
1) optimise_portfolio(...)      : main monthly backtest loop and data plumbing
2) solve_pf_optimisation(...)   : objective, constraints, and PyTorch optimiser
3) build_portfolio_dataframe(...) and shrink_universe(...) : data alignment rules
4) load_portfolio_backtest_data(...) and load_MLpredictions(...) : I/O and inputs
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3
import pickle

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

#Gradient Ascent
import torch
import torch.nn.functional as F

import statsmodels.api as sm

#Saving results
from copy import deepcopy
import json
import re

#Custom Functions
import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Functions: Portfolio Selection

def get_universe_partitions(prev_date, date, df_pf_weights):
    """
    Partition the stock universe at trading date t
    into stayers, leavers, newcomers and helpers.

    * ``stayers``  : stocks that remain in the S&P500.
    * ``leavers``  : stocks that leave the S&P500 (must be fully liquidated --> π_t = 0).
    * ``newcomers``: stocks newly added to the S&P500 (no previous portfolio weight available)
    * ``active``   : sorted list of stocks that are candidates for non-zero
                     portfolio weights at the trading date
    * ``zeros``    : Stocks that must have π_t = 0; starts as
                     the list of leavers and may be extended later (e.g. for
                     missing Kyle's lambda or missing covariance/return data).

    """
    
    #Stock universes
    prev_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == prev_date]['id'])
    cur_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == date]['id'])
    
    #Stocks that can no longer be in the portfolio
    leavers = list(prev_universe - cur_universe)
    #Stocks that can newly enter the portfolio
    newcomers = list(cur_universe - prev_universe)
    #Stocks that can remain the portfolio
    stayers = list(cur_universe.intersection(prev_universe))
    #Stocks which can have non-zero portfolio weights and are active choice variables
    active = sorted(list(set(newcomers + stayers)))
    
    #Stocks for which pi_t = 0 must be enforced. This affects all leavers.
    #On top of that, it can affect a subset of newcomers due to missing data 
    #on Kyle's Lambda or covariance-matrix
    zeros = leavers.copy()
    
    return stayers, leavers, newcomers, active, zeros

def optimise_portfolio(
    df_pf_weights: pd.DataFrame,
    df_kl: pd.DataFrame,
    df_me: pd.DataFrame,
    dict_barra: dict,
    df_returns: pd.DataFrame,
    df_wealth: pd.DataFrame,
    df_spy: pd.DataFrame,
    df_retPred: pd.DataFrame,
    trading_dates,
    prediction_col: str ,
    flat_MaxPi: bool,
    flat_MaxPi_limit: float,
    w_upperLimit: float,
    w_lowerLimit: float,
    vol_scaler: float,
    tc_scaler: float,
) -> pd.DataFrame:
    
    """
    
    Solves the one-period myopic portfolio problem for each trading date.

    For each month t in ``trading_dates``, this function:

    1. Determines the investable universe (stayers, leavers, newcomers).
        Stock with with missing values for any of Kyle's lambda, covariance, or predictions
        must be completely sold off.
    2. Builds the Barra-style covariance matrix Σ_t for the active universe.
    3. Assembles all required inputs (estimated returns, Kyle's lambda,
       lagged portfolio weights, benchmark variance, AUM).
    4. Solves the myopic portfolio optimisation via
       gradient ascent in PyTorch.

    5. Stores the resulting weights, realised revenue and transaction costs.

    Args:
        df_pf_weights: DataFrame of current portfolio weights π_t for each
            stock and date. Must contain ``['id', 'eom', 'pi', 'g']`` where
            ``g`` is the growth factor g_t^w used to compute the
            drifted portfolio weights ( G_t π_{t-1} ).
        df_kl: Kyle's lambda data. Requires columns ``['id', 'eom', 'lambda']``.
        df_me: Market equity / size data. Used to construct upper (and lower)
            bounds on portfolio weights relative to benchmark weights.
        dict_barra: Dictionary mapping dates to Barra covariance matrix.
        df_returns: DataFrame containing realised stock returns, used to
            compute realised revenue; columns ``['id', 'eom', 'tr']``.
        df_wealth: DataFrame of AUM (wealth) per date; columns
            ``['eom', 'wealth']``.
        df_spy: Benchmark (SPY) data with at least columns
            ``['eom', 'variance']`` giving benchmark return variance per month.
        df_retPred: DataFrame with model-based expected returns. Must include
            ``['id', 'eom', prediction_col]`` for at least the trading dates.
            Note that prediction for trading date t is made at date t-1.
        trading_dates: Iterable of end-of-month timestamps indicating the
            portfolio rebalancing dates.
        prediction_col: Column name in ``df_retPred`` containing the expected
            (excess) returns used in the objective.
        w_upperLimit: Scalar multiple applied to benchmark weights to obtain
            the maximum allowed portfolio weight for each stock.
        w_lowerLimit: Scalar multiple applied to benchmark weights to obtain
            the minimum allowed portfolio weight for each stock.
        vol_scaler: Scaling factor applied to the benchmark variance constraint
            σ_B^2; >1 relaxes, <1 tightens the variance limit.
        tc_scaler: Scaling factor applied to Kyle's lambda to run
            sensitivity analyses on transaction costs.

    Returns:
        pd.DataFrame: Long dataframe with one row per (id, eom) containing:

            - optimal portfolio weight π_t
            - Kyle's lambda
            - lagged G_t π_{t-1} (pi_g_tm1)
            - realised end-of-period return ``tr``
            - Realised revenue ``rev = π_t * tr``
            - transaction cost ``tc``

        The returned dataframe aggregates the strategy for all trading dates.
    """

    # Ensure for each trading date there is a Return Prediction
    # Intersect trading dates with available prediction timestamps and add one extra terminal month so the last available forecast can be traded.
    trading_dates = list(sorted(set(trading_dates).intersection(set(df_retPred.eom)).copy()))
    #Add one more date as for trading date t, the predictions are based on date t-1
    trading_dates = trading_dates + [trading_dates[-1] + pd.offsets.MonthEnd(1)]

    #Container to store results
    df_strategy = []    
    
    # Scale Transaction costs
    df_kl = df_kl.copy() #avoid overwriting the function input
    df_kl['lambda'] = df_kl['lambda']*tc_scaler

    #Loop over trading date
    for date in trading_dates:
        print(date)

        #=====================================
        #             Preliminaries
        #=====================================
        
        #Previous Date (the information we possess)
        prev_date = date - pd.offsets.MonthEnd(1)
        
        #Stock universe
        stayers, leavers, newcomers, \
            active, zeros = get_universe_partitions(prev_date, date, df_pf_weights)
            
        #Compute Barra covariance matrix
        Sigma = GF.create_cov(dict_barra[prev_date])
        
        #Get return predictions
        df_return_predictions = (df_retPred
                              .loc[df_retPred['eom'] == prev_date]
                              .get(['id', 'eom', prediction_col])
                              )
        
        #=====================================================
        #   Shrink active universe in case of missing data
        #=====================================================
        
        active, zeros = shrink_universe(prev_date, active, newcomers, zeros,
                            df_kl, Sigma, df_return_predictions)
        
        #========================================
        #   Reduce DataFrames to active universe 
        #========================================
        
        Sigma_active, df_kl_active, df_ret_pred_active = reduce_to_active(prev_date, active, Sigma, df_kl, df_return_predictions)
        
        # Save Workspace
        del Sigma
        
        #==========================================================
        #   Build DataFrame for Portfolio Optimisation
        #==========================================================
        
        df_pf_t = build_portfolio_dataframe(date, prev_date,
                                      active, stayers, newcomers, leavers, zeros,
                                      df_pf_weights, df_kl, df_returns,
                                      df_me)
        
        #==========================================================
        #           Solve for optimal portfolio
        #==========================================================
        
        # Solve for optimal portfolio weight for stocks in the active universe
        pi_opt = \
            solve_pf_optimisation(prev_date, date, 
                                  active, 
                                  df_pf_t, df_kl_active, df_ret_pred_active, Sigma_active, 
                                  df_me, df_spy, df_wealth, prediction_col,
                                  flat_MaxPi, flat_MaxPi_limit,
                                  w_upperLimit, w_lowerLimit, vol_scaler, tc_scaler)
        
        # Convert to Series
        pi_opt = pd.Series(pi_opt, index=active, name='pi_opt')
        
        # Merge to Dataframe
        df_pf_t = df_pf_t.merge(pi_opt, left_on='id', right_index=True, how='left')
        
        # Overwrite previous values
        df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = df_pf_t['pi_opt']
        
        # Drop auxiliary column
        df_pf_t = df_pf_t.drop(columns='pi_opt')
        
        # print pi_max value
        print(f"   MAX pi: {df_pf_t['pi'].max()}") 
        
        #==========================================================
        #           Compute Revenue & TC
        #==========================================================
        
        # Get Begin of Period 'date' Baseline wealth level
        w = df_wealth[df_wealth['eom'] == prev_date]['wealth'].iloc[0]
        
        # Compute revenue & transaction costs (pi is begin of month, 'tr' is end of month)
        df_pf_t = (df_pf_t
                .assign(rev = lambda df: df['pi']*df['tr'])
                # Important: Transaction cost scale (tc_scaler) is absorbed into df['lambda']
                .assign(tc = lambda df: (df['pi'] - df['pi_g_tm1'])**2 * df['lambda'] * float(w)/2 )
                )
        
        # Update DataFrame storing Portfolio Weights (will be used throughout to compute 'pi_g_tm1')
        df_pf_weights = df_pf_weights.set_index(['id', 'eom'])
        df_pf_weights.update(df_pf_t.set_index(['id', 'eom'])[['pi']])
        df_pf_weights = df_pf_weights.reset_index()
        
        #Append Result
        df_strategy.append(df_pf_t)
        
    #Make DataFrame of strategy
    df_strategy = pd.concat(df_strategy)
    
    return df_strategy, df_pf_weights


def shrink_universe(prev_date, active, newcomers, zeros,
                    df_kl, Sigma, return_predictions):
    """
    Shrink the active stock universe based on data availability.

    This function enforces that every stock in the active universe has:
    - Kyle's lambda at ``prev_date``
    - a finite variance entry in the Barra covariance matrix ``Sigma``
    - a return prediction for ``prev_date``.

    Any stock that fails one of these checks is added to ``zeros``
    (i.e. forced to have π_t = 0), and removed from the ``active`` set.

    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous end-of-month date (t-1), i.e. the information date.
    active : list of hashable
        List of stock ids that are candidates for non-zero portfolio
        weights at date t.
    newcomers : list of hashable
        Stocks that newly enter the investable universe at date t.
    zeros : list of hashable
        Stocks that must have π_t = 0. Initially contains leavers and
        is extended in this function.
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    Sigma : pandas.DataFrame
        Full Barra covariance matrix for ``prev_date`` with stock ids
        as both index and columns.
    return_predictions : pandas.DataFrame
        DataFrame with at least columns ``['id', 'eom', prediction_col]``
        for ``eom == prev_date``.

    Returns
    -------
    active : list
        Sorted list of stock ids that remain eligible for non-zero
        weights after all data-availability checks.
    zeros : list
        Updated list of stock ids that must have π_t = 0.
    """
    
    #--- Kyle's lambda ---
    kl_prev = df_kl.loc[df_kl['eom'] == prev_date, 'id']
    zeros.extend([s for s in newcomers if s not in set(kl_prev)])
    active = sorted([s for s in active if s not in zeros])

    #--- Barra Covariance ---
    Sigma = Sigma.reindex(index=active, columns=active).copy()

    zeros.extend(list(Sigma.index[pd.isna(np.diag(Sigma))]))
    active = sorted([s for s in active if s not in zeros])
    
    # --- Return predictions ---
    zeros.extend([s for s in active if s not in return_predictions['id'].values])
    active = sorted([s for s in active if s not in zeros])

    return active, zeros

def reduce_to_active(prev_date, active, Sigma, df_kl, return_predictions):
    """
    Restrict inputs to the active stock universe and sort by id.
    
    This function:
    1. Slices the Barra covariance matrix ``Sigma`` to the active ids.
    2. Extracts Kyle's lambda for the active ids at ``prev_date``.
    3. Restricts return predictions to the active ids.
    
    All returned objects are sorted by ``id`` so that they are aligned
    for vectorised operations in PyTorch.
    
    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous end-of-month date (t-1).
    active : list of hashable
        List of stock ids that are eligible for non-zero portfolio
        weights at date t.
    Sigma : pandas.DataFrame
        Full Barra covariance matrix (index and columns are stock ids).
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    return_predictions : pandas.DataFrame
        DataFrame with columns ``['id', 'eom', prediction_col]`` for
        at least ``eom == prev_date``.
    
    Returns
    -------
    Sigma_active : pandas.DataFrame
        Covariance matrix restricted to the active universe, with both
        index and columns equal to ``active`` (sorted).
    df_kl_active : pandas.DataFrame
        Kyle's lambda for active stocks at ``prev_date``, sorted by id.
    return_predictions_active : pandas.DataFrame
        Return predictions for active stocks at ``prev_date``,
        sorted by id.
    """
    
    #Covariance Matrix (sorted by 'id' as active is sorted)
    Sigma = Sigma.loc[active, active]

    kyles_lambda = (df_kl
                    .loc[(df_kl['eom'] == prev_date) & (df_kl['id'].isin(active))]
                    .sort_values('id')
                    .reset_index(drop=True))

    return_predictions = (return_predictions
                          .loc[return_predictions['id'].isin(active)]
                          .sort_values('id')
                          .reset_index(drop=True))

    return Sigma, kyles_lambda, return_predictions

def build_portfolio_dataframe(date, prev_date,
                              active, stayers, newcomers, leavers, zeros,
                              df_pf_weights, df_kl, df_returns,
                              df_me):
    """
    Build the per-period portfolio dataframe used in optimisation.
    
    Constructs a DataFrame with one row per stock in the union of
    stayers, newcomers and leavers at trading date ``date``. It
    initialises portfolio weights, merges Kyle's lambda, realised
    returns, and computes the drifted portfolio weights G_t π_{t-1}.
    
    Parameters
    ----------
    date : pandas.Timestamp
        Current trading date (end-of-month t).
    prev_date : pandas.Timestamp
        Previous trading date (t-1).
    stayers, newcomers, leavers : list
        Universe partitions returned by ``get_universe_partitions``.
    zeros : list
        Stocks that are known to have π_t = 0 (e.g. leavers and stocks
        with missing data).
    df_pf_weights : pandas.DataFrame
        Historical portfolio weights with columns
        ``['id', 'eom', 'pi', 'g']`` at least.
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    df_returns : pandas.DataFrame
        Realised returns with columns ``['id', 'eom', 'tr']``.
    
    Returns
    -------
    df_portfolio_t : pandas.DataFrame
        Portfolio state at date t with columns including
        ``['id', 'eom', 'pi', 'lambda', 'tr', 'pi_g_tm1']``.
        The column ``pi`` is initialised to a small positive value
        (1e-16) and will be overwritten by the optimiser for active
        stocks.
    """
    
    #---- Initialisation ----
    df_pf_t = (pd.DataFrame({ #df_portfolio_t
        'id': list(stayers + newcomers + leavers),
        'eom': date,
        'pi': np.array(1e-4)
    }).sort_values(by = 'id').reset_index(drop=True))

    # ---- Merge Kyle's lambda  ----
    #   (Note, need KL also for leavers and not just for active)
    df_pf_t = df_pf_t.merge(df_kl[df_kl['eom'] == prev_date][['id', 'lambda']],
                            on='id', how='left')
    
    #Set 'lambda' to zero for newcomers for which 'lambda' is NA. 
    #   Reason: When computing transaction costs, valid values for 'lambda' are
    #           required. If newcomers have missing 'lambda', they will not be
    #           be traded (i.e. they are not in active). Thus, their portfolio 
    #           weight is and will remain zero, so no transaction costs will be 
    #           incured anyway.
    df_pf_t.loc[(df_pf_t['id'].isin(set(newcomers).intersection(set(zeros)))) 
                &
                (df_pf_t['lambda'].isna()), 'lambda'] = 0.0
    
    if df_pf_t['lambda'].isna().sum() > 0:
        print("ERROR: A stock does not have a value for Kyle's Lambda")

    # ---- Realised Returns (to compute profit later) ---
    # Merge realised return
    df_pf_t = df_pf_t.merge(df_returns[df_returns['eom'] == date][['id', 'tr']],
                            on='id', how='left')
    
    #Set return for leavers to zero to avoid NaNs spreading, i.e. pi * tr = 0*NaN = NaN.
    df_pf_t.loc[df_pf_t['id'].isin(leavers), 'tr'] = 0

    # ---- Drifted Weights G @ pi_{t-1} ----
    # Compute
    pi_g = (df_pf_weights.query("eom == @prev_date")[['id', 'pi', 'g']]
            .assign(pi_g=lambda df: df['pi'] * df['g'])
            [['id', 'pi_g']])
    
    # Merge
    df_pf_t = df_pf_t.merge(pi_g, on='id', how='left').rename(columns={'pi_g': 'pi_g_tm1'})
    
    #Set value for newcomers to zero
    df_pf_t.loc[df_pf_t['id'].isin(newcomers), 'pi_g_tm1'] = 0
    
    # ---- Initialise 'pi_t' with 'pi_g_tm1' ----
    
    # Initialise pi_t with G @ pi_{t-1}
    df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = df_pf_t.loc[df_pf_t['id'].isin(active), 'pi_g_tm1']
    
    # For newcomers that are actively traded --> g pi_{t-1} = 0.
    #   So, set pi_t to some epsilon (else-wise log(pi_t) undefined)
    df_pf_t.loc[(df_pf_t['pi'] == 0.0) & df_pf_t['id'].isin(active), 'pi'] = 1e-4
    
    # Set any 'pi' for zeros to 0.0
    df_pf_t.loc[df_pf_t['id'].isin(zeros), 'pi'] = 0.0

    return df_pf_t

def solve_pf_optimisation(prev_date, date, 
                          active, 
                          df_pf_t, df_kl_active, df_ret_pred_active, Sigma_active, 
                          df_me, df_spy, df_wealth, prediction_col,
                          flat_MaxPi, flat_MaxPi_limit,
                          w_upperLimit, w_lowerLimit, vol_scaler, tc_scaler):
    
    """
    Solve the one-period myopic portfolio optimisation for date t.
    
    The optimiser chooses portfolio weights π_t over the active universe
    to maximise expected revenue minus transaction costs, subject to:
    
    - portfolio weights summing to one (via softmax parametrisation),
    - upper and lower bounds on each π_t,i defined as multiples of
      benchmark (value) weights from ``df_me``,
    - a maximum allowed portfolio variance scaled by ``vol_scaler``.
    
    Transaction costs are modelled as quadratic in turnover using
    Kyle's lambda
    
    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous trading date (t-1).
    date : pandas.Timestamp
        Current trading date (t).
    active : list
        List of stock ids in the active universe at date t.
    df_pf_t : pandas.DataFrame
        Portfolio dataframe at date t as returned by
        ``build_portfolio_dataframe``. Must contain columns
        ``['id', 'pi', 'pi_g_tm1']`` at least.
    df_kl_active : pandas.DataFrame
        Kyle's lambda for active stocks at ``prev_date`` with columns
        ``['id', 'lambda']``.
    df_return_predictions_active : pandas.DataFrame
        Expected returns for active stocks at ``prev_date`` with
        columns ``['id', prediction_col]``.
    Sigma : pandas.DataFrame
        Covariance matrix for active stocks at ``prev_date``.
    df_me : pandas.DataFrame
        Market equity data with columns ``['id', 'eom', 'me']`` used to
        derive weight bounds.
    df_spy : pandas.DataFrame
        Benchmark variance data with columns ``['eom', 'variance']``.
    df_wealth : pandas.DataFrame
        Wealth (AUM) data with columns ``['eom', 'wealth']``.
    prediction_col : str
        Name of the expected return column in
        ``df_return_predictions_active``.
    w_upperLimit, w_lowerLimit : float
        Multipliers applied to value weights to derive upper and lower
        bounds on π_t,i.
    vol_scaler : float
        Multiplier applied to benchmark variance constraint.
    
    Returns
    -------
    numpy.ndarray
        Array of optimised portfolio weights π_t for the active stocks,
        ordered consistently with the rows of ``df_pf_t`` restricted to
        ``active`` and with ``Sigma`` / ``df_return_predictions_active``.
    """

    # ---- Define Torch Objects ----
    # Return predictions
    r = torch.tensor(df_ret_pred_active[prediction_col], dtype=torch.float64)
    
    # Covariance Matrix 
    S = torch.tensor(Sigma_active.to_numpy(), dtype=torch.float64)
    
    # Kyle's Lambda (diagonal) Matrix
    L_diag = torch.tensor(df_kl_active['lambda'], dtype=torch.float64)

    # Wealth (AUM) Begin of period 'date'
    w = torch.tensor(df_wealth[df_wealth['eom'] == prev_date]['wealth'].iloc[0], dtype=torch.float64)
    
    # Drifted portfolio weights G @ pi_{t-1}
    pi_g_tm1 = torch.tensor(df_pf_t[df_pf_t['id'].isin(active)]['pi_g_tm1'].to_numpy(),
                            dtype=torch.float64)
    
    # Logits of pi_t
    pi_logits = torch.tensor(
        np.log(df_pf_t[df_pf_t['id'].isin(active)]['pi'].to_numpy()),
        requires_grad=True,
        dtype=torch.float64
    )
    
    # ---- Bound on pi_t ----
    
    if flat_MaxPi:
        max_pi = torch.tensor(flat_MaxPi_limit, dtype=torch.float64)
        min_pi = torch.tensor(0.0, dtype=torch.float64)
        
    else:
        weights_df = (df_me
                      .loc[(df_me['eom'] == prev_date) & (df_me['id'].isin(active))]
                      .sort_values(by = 'id')
                      .assign(w_max=lambda df: df['me'] / df['me'].sum() * w_upperLimit)
                      .assign(w_min=lambda df: df['me'] / df['me'].sum() * w_lowerLimit))
        
        max_pi = torch.tensor(weights_df['w_max'].to_numpy(), dtype=torch.float64)
        min_pi = torch.tensor(weights_df['w_min'].to_numpy(), dtype=torch.float64)

    # ---- Bounds on portfolio variance ----
    max_var = df_spy[df_spy['eom'] == prev_date]['variance'].iloc[0]*vol_scaler
    
    # --- Scaling Inequality Constraints ----
    penalty_maxPi   = 1.0
    penalty_minPi   = 1.0
    penalty_var     = 1.0

    # ---- Optimizer ----
    optimizer = torch.optim.Adam([pi_logits], lr=1e-2)

    # ---- Gradient Ascent ----
    for _ in range(500):
        # Clear Gradient
        optimizer.zero_grad()
        
        # pi_t (in levels)
        pi = F.softmax(pi_logits, dim=0)
        
        # Predicted revenue
        revenue = torch.dot(r, pi)
        
        # Transaction costs
        diff = pi - pi_g_tm1
        tc = 0.5 * w * (L_diag * diff.pow(2)).sum()
        
        # pi_t bounds violation
        max_pi_violation = (penalty_maxPi * F.relu(pi - max_pi)).sum()
        min_pi_violation = (penalty_minPi * F.relu(min_pi - pi)).sum()
        
        # Variance Violation  
        var_violation = penalty_var * F.relu(pi @ S @ pi - max_var)

        # Loss Function
        F_val = revenue - tc
        loss = -F_val + max_pi_violation + min_pi_violation + var_violation
        
        # One step of gradient ascent
        loss.backward()
        optimizer.step()

    # Print predicted profit
    print(f"  Predicted Profit: {F_val.item()}")
    
    return pi.detach().cpu().numpy()

#%% Function: Saving Output
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

def settings_to_id(settings: dict) -> str:
    """
    File-system friendly ID based on settings (used in filenames).
    """
    s = json.dumps(settings, sort_keys=True)
    # Replace anything non-alphanumeric with underscores
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s


#%% Function: Reading in Data
def load_portfolio_backtest_data(con, start_date, sp500_ids, path):
    """
    Load and assemble all inputs required for the portfolio backtest.

    This function pulls data for the investable universe (S&P 500 subset),
    Kyle's lambda, market equity, realised returns, the exogenous AUM
    evolution, and the Barra-style covariance matrices. It also constructs
    the initial portfolio weights (value-weighted) 
    and the exogenous AUM growth factor `g_t`
    used in the portfolio optimisation.

    Parameters
    ----------
    con : sqlite3.Connection
        Open SQLite connection to the JKP/Factors database that contains
        the table ``Factors_processed``.
    start_date : str or pandas.Timestamp
        First end-of-month date (inclusive) for which the backtest should
        be run. Data for the portfolio initialisation is pulled from
        ``start_date - 1 month`` as well.
    sp500_ids : str
        String of comma-separated stock identifiers (e.g. PERMNOs) used in
        the SQL ``IN (...)`` clause to restrict the universe to S&P 500
        constituents over time.
    path : str or pathlib.Path
        Root path to the project directory that contains the
        ``Data/`` subdirectory. Used to locate
        ``wealth_evolution.csv`` and ``Barra_Cov.pkl``.
        
    Returns
    -------
    df_pf_weights : pandas.DataFrame
        DataFrame of portfolio weights and AUM growth factors with columns:

        - ``id`` : stock identifier of S&P500 stocks ONLY!
        - ``eom`` : end-of-month timestamp
        - ``pi`` : portfolio weight at the beginning of the month;
          initial month is value-weighted by market equity, subsequent
          months are initialised with a small positive value (``1e-16``)
        - ``g`` : exogenous AUM growth factor g_t^w used to construct
          G_t π_{t-1}. For the first month, ``g = 1``; thereafter,
          ``g = (1 + tr) / (1 + mu)``, where ``tr`` is the stock return
          and ``mu`` is the benchmark/market return.

    df_kl : pandas.DataFrame
        Kyle's lambda (price impact) data with columns:

        - ``id``
        - ``eom``
        - ``lambda`` : Kyle's lambda at the beginning of each month.

    df_me : pandas.DataFrame
        Market equity (size) data with columns:

        - ``id``
        - ``eom``
        - ``me`` : market equity at the beginning of each month.

    df_returns : pandas.DataFrame
        Realised return data used for evaluation.

    df_wealth : pandas.DataFrame
        Exogenous AUM evolution with at least columns:

        - ``eom`` : end-of-month timestamp
        - ``wealth`` : assets under management at the beginning of month t
        - ``mu`` : market / benchmark return used to compute g_t^w

        Only rows with ``eom >= start_date - 1 month`` are kept.

    dict_barra : dict
        Dictionary mapping end-of-month dates (keys) to Barra-style
        covariance model objects (values). Only entries with date
        ``>= start_date`` are retained. Each value is expected to be
        consumable by ``GF.create_cov(dict_barra[date])`` to produce
        the stock-level covariance matrix Σ_t.

    """
    #---- Data for the investable universe ----
    query = ( "SELECT id, eom, in_sp500, me, lambda, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1 "
             + "FROM Factors_processed "
             + f"WHERE eom >= '{start_date}' "
             # loads every stock that ever was in the S&P 500
             # (loads data more quickly, active S&P500 universe restriction is ensured by
             # using 'in_sp500' == True. This is done shortly. 
             + f"AND id IN ({sp500_ids})")

    df = (pd.read_sql_query(query,
                           parse_dates = {'eom'},
                           con=con
                           )
          .sort_values(by = ['eom', 'id'])
          .assign(in_sp500 = lambda df: df['in_sp500'].astype('boolean'))
          )

    #---- Subset Kyle's Lambda ----
    df_kl = df.get(['id', 'eom', 'lambda'])
    
    #---- Subset Market Equity ----
    df_me = df.get(['id', 'eom', 'me'])
    
    #---- Subset Realised Returns ----
    df_returns = df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1'])

    #---- Evolution AUM ----
    df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
    df_wealth = df_wealth.loc[df_wealth['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)]

    #---- Initialise DataFrame for Portfolio Weights ----
    # Restrict to active S&P500 stocks
    df_pf_weights = df.loc[df['in_sp500']].get(['id','eom','me', 'tr']) 

    #Compute initial value weighted portfolio
    df_pf_weights = (
        df_pf_weights
        # 1. Only include relevant dates
        .pipe(lambda df: df.loc[df['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)])
        # 2. Calculate the aggregate market cap per date
        .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
        # 3. Calculate a value-weighted initial portfolio
        .assign(pi=lambda df: df['me'] / df['group_sum'])
        # 4. Set all portfolio weights to zero if 'eom' > min_date (these are just placeholders)
        .assign(pi=lambda df: np.where(df['eom'] > df['eom'].min(), 1e-16, df['pi']))
        # 5. Calculate 'g'
        .merge(df_wealth[['eom', 'mu']], on=['eom'], how='left') # Get Market Return
        .assign(
            is_min_eom=lambda df: df['eom'] == df['eom'].min(),
            g=lambda df: np.where(
                df['is_min_eom'],
                1,
                (1 + df['tr']) / (1 + df['mu'])
            )
        )
        # 6. Clean up
        .drop(columns=['group_sum', 'is_min_eom', 'mu', 'me', 'tr'])
        .sort_values(by = ['eom','id'],ascending = [True,True])   
    )

    #---- Barra Covariance Matrix ----
    # Load Barra covariance
    with open(path + "Data/Barra_Cov.pkl", "rb") as f:
        dict_barra_all = pickle.load(f)
        
    dict_barra = {
        k: v for k, v in dict_barra_all.items() 
        if k >= pd.to_datetime(start_date)
    }

    print("Data loading complete.")
    
    return df_pf_weights, df_kl, df_me, df_returns, df_wealth, dict_barra

def load_MLpredictions(DataBase, predictions:list):
    """
    Load and merge ML model return predictions.

    All ML models from ``ensemble``, are merged into one DataFrame.

    Note: At date ``eom``, the predictions are for the return at ``eom`` +1

    Only rows with non-missing values across all ensemble members
    are retained (rows with any NaNs are dropped).
    """
    
    #Initialise Dataframe 
    df = None
    
    #Loop over models in ensemble
    for model in predictions:
        #Load Predictions
        df_next = pd.read_sql_query(f"SELECT * FROM {model}", con=DataBase, parse_dates={"eom"})
        if df is None:
            df = df_next
        else:
            #Merge to existing dataframe
            df = df.merge(df_next, on=['id','eom'], how='outer')
    
    return df

#%% Read in Data

#DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_processed.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")
Models = sqlite3.connect(database = path + "Data/Predictions.db")

#============================
#       Trading Dates
#============================
#Trading dates
trading_start, trading_end = pd.to_datetime("2004-01-31"), pd.to_datetime("2024-12-31")
trading_dates = pd.date_range(start=trading_start,
                              end=trading_end,
                              freq='ME'
                              )

#Start and End Date as Strings
start_date = str(trading_start - pd.offsets.MonthEnd(1))[:10]
end_date = str(trading_end)[:10]

#===============================
#       SPY ETF Performance
#===============================
# The historical variance is required for the volatility benchmarking
df_spy = pd.read_sql_query("SELECT * FROM SPY",
                           parse_dates = {'eom'},
                           con = Benchmarks)

#============================
#       S&P 500 Universe
#============================
#Stocks that were, are and will be in the S&P 500.
sp500_ids = list(pd.read_sql_query("SELECT * FROM SP500_Constituents_alltime",
                              con = SP500_Constituents)['id']
                 )
sp500_ids = ', '.join(str(x) for x in sp500_ids)

#Stocks that at date t are in the S&P 500
sp500_constituents = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )

#===============
#       Data
#===============
df_pf_weights, df_kl, df_me, df_returns,\
    df_wealth, dict_barra = load_portfolio_backtest_data(JKP_Factors, start_date, 
                                          sp500_ids, path)
    
#============================
#      Model Predictions 
#============================

# List of estimators to compute the portfolios for
estimators = [
          #'XGBRegHPlenient_LevelTrMsp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          #'TransformerSet_Dropout010_LevelTrMSp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          #'RFF_LevelTrMsp500Target_SP500UniverseFL_ZscoreFeatures_RollingWindow_win120_val12_test12',
          'IPCA_LevelTrMsp500Target_CRSPUniverse_ZscoreFeatures_RollingWindow_win120_val12_test12',
          ]

#Load return predictions
#At 'eom', predictions are for eom+1
df_retPred = load_MLpredictions(Models, estimators) 

#Get names of return predictors
prediction_cols = list(df_retPred.columns.drop(['id','eom']).astype('string'))

df_retPred = df_retPred.dropna().reset_index(drop = True)

#Update prediction_cols name
prediction_cols = list(df_retPred.columns.drop(['id','eom']).astype('string'))

#%% Compute Optimal Portfolio

for tc_scale in [1.0, 0.5, 0.1, 0.01]: # Loop over different transaction cost regimes
    
    #---- Common Settings for This Run (same for all models) ----
    run_settings = dict(includeRF    = False, # Outside asset not implemented
                        flatMaxPi    = True,
                        flatMaxPiVal = 1.0,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = 1.0, 
                        tcScaler     = tc_scale,
                        )
    
    print("Run ID:", settings_to_id(run_settings))
    print("Settings:", settings_string(run_settings))
    
    #---- Save empty portfolio weights ----
    df_pf_weights_base = df_pf_weights.copy()
    
    #---- Containers to collect results across models ----
    per_model_returns   = {}   # model_name -> df_ret_strat
    per_model_strategy  = {}   # model_name -> df_strategy
    per_model_metrics   = []   # list of dicts with metrics
    
    #---- Benchmark cumulative returns ----
    df_spy_run = (df_spy[df_spy['eom'].between(trading_start, trading_end)]
                  .assign(cumulative_return = lambda df: (1.0 + df['ret']).cumprod()
                          )
                  )
        
    for model_name, prediction_col in zip(estimators, prediction_cols):
        print("\n==============================")
        print("Running model:", model_name)
        print("    Prediction column:", prediction_col)
        print("==============================")
        
        # Use a fresh copy of the initial weights for each model
        df_pf_weights = df_pf_weights_base.copy()
        
        #---- Compute Optimal Portfolio ----
        df_strategy, df_pf_weights \
            = optimise_portfolio(df_pf_weights, df_kl, df_me, dict_barra, df_returns, df_wealth, df_spy, 
                               df_retPred, 
                               trading_dates, 
                               prediction_col, 
                               run_settings['flatMaxPi'], 
                               run_settings['flatMaxPiVal'], #portfolio weight bound  [0,flat_MaxPi_limit] for every stock
                               run_settings['Wmax'], 
                               run_settings['Wmin'], #Benchmark dependent portfolio bound for every stock
                               run_settings['volScaler'],
                               run_settings['tcScaler'],
                               )
            
        # Merge predicted returns
        df_strategy = df_strategy.merge(df_retPred.get(['id','eom',prediction_col])
                                        .assign(eom = lambda df: df['eom'] + pd.offsets.MonthEnd(1)),
                                        on = ['id', 'eom'], how = 'left')  
        
        # ---- Compute strategy returns (net & gross) per month ----
            
        # Compute Profit
        df_ret_strat = (
            df_strategy
            .assign(ret_net_row = lambda df: df['rev'] - df['tc'])
            .groupby('eom', as_index=False)
            .agg(
                ret_net   = ('ret_net_row', 'sum'),
                ret_gross = ('rev', 'sum')
            )
            .assign(
                cumret_net   = lambda df: (1.0 + df['ret_net']).cumprod(),
                cumret_gross = lambda df: (1.0 + df['ret_gross']).cumprod()
            )
        )
        
        # Compute cumulative monthly profit for benchmark
        df_ret_bench = (df_spy[df_spy['eom'].isin(df_ret_strat['eom'].unique())]
                               .assign(cumulative_return = lambda df: (1+df['ret']).cumprod())
                               .reset_index(drop = True)
                               )
            
        # ---- Save per-model pickle (with settings attached) ----
        result_dict = {
        'run_settings' : run_settings,
        'model_name'   : model_name,
        'prediction_col': prediction_col,
        'Strategy'     : df_strategy,
        'Profit'       : df_ret_strat,
        }
    
        safe_model_name = re.sub(r"[^0-9a-zA-Z]+", "_", model_name).strip("_")
        
        filename = f"{settings_string(run_settings)}_{safe_model_name}.pkl"
        
        with open(os.path.join(path, "Portfolios", filename), "wb") as f:
            pickle.dump(result_dict, f)
        
        print(f"Saved model results to: {filename}")