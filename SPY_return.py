# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:21:18 2025

@author: Patrick

Computes the return of the SPDR (Standard & Poor's Depositary Receipt) ETF) of
S&P 500 as a benchmark

Can also compare against other ETFs:
    Momentum: iShares MSCI USA Momentum Factor ETF (MTUM)
    
    Quality: iShares MSCI USA Quality Factor ETF (QUAL)
    
    Minimum Volatility: iShares MSCI Min Vol USA ETF (USMV)

Other benchmark:
    Jensen's Alpha:
    
    This is the "gold standard" for proving alpha. You run a regression of your portfolio's excess returns against the market's excess returns:
    (πt′​rt+1​−rf,t+1​)=α+β(SP500TRt+1​−rf,t+1​)+ϵt​
    
    Your Goal: You want a positive and statistically significant α (alpha). This proves you generated returns after accounting for all the market risk (beta) you took.
"""
#%% Libraries
import yfinance as yf
import pandas as pd
import sqlite3

import General_Functions as GF

path = "C:/Users/patri/Desktop/ML/"

settings = GF.get_settings()

#%% SPY S&P 500 Returns

Benchmarks = sqlite3.connect(database=path +"Data/Benchmarks.db")


#---- Download Data ----
spyticker = yf.Ticker("^SP500TR") #Total Return (Dividends reinvested)
#'Close' is adjusted close due to auto_adjust = True
df_spy = spyticker.history(period="max", interval="1d", start="1980-01-01", end="2024-12-31", 
                              auto_adjust=True, rounding=True)

# ---- Reformat Date ----
#Get rid of hours, minutes,...
df_spy.index = df_spy.index.tz_localize(None).normalize()

#===============================
#       Monthly Returns
#===============================
#Fill missing days (non-trading days) with values from last trading day
df_spy_m = (df_spy.reindex(pd.date_range(start=df_spy.index.min(), 
                                      end=df_spy.index.max() + pd.offsets.MonthEnd(0), 
                                      freq='D'))
          .ffill()
          .reset_index()
          .rename(columns={'index': 'Date'})
          )

#Keep end of month close to match frequency of other data
df_spy_m = df_spy_m.loc[df_spy_m['Date']
                    .isin(pd.date_range(start = df_spy_m['Date'].min(), 
                                        end = df_spy_m['Date'].max()+pd.offsets.MonthEnd(0), 
                                        freq = "ME")
                          )
                    ]

# ---- Compute return ----
monthly_returns = df_spy_m['Close'].pct_change().dropna().to_frame().reset_index(drop=True)
monthly_returns['eom'] = pd.date_range(start = df_spy_m['Date'].min(), 
                                       end = df_spy_m['Date'].max(), 
                                       freq = "ME")[1:]

monthly_returns = monthly_returns.rename(columns = {'Close':'ret'})


#=================================
#     Monthly Return Volatility
#=================================

# Calculate daily returns first
df_spy['daily_ret'] = df_spy['Close'].pct_change()

# Drop the first NaN row
df_spy = df_spy.dropna(subset=['daily_ret'])

# Half-life of weight decay
hl_var = int(252 / 2) # 126 days

#Compute the EWMA for the daily return variance
df_spy['daily_var_ewm'] = df_spy['daily_ret'].ewm(
    halflife=hl_var, 
    min_periods=hl_var
).var()

#Only get the last period of the month
df_spy_var = df_spy['daily_var_ewm'].resample('ME').last().dropna()

#Adjust to monthly
df_spy_var = (df_spy_var * 21).to_frame()
df_spy_var.columns = ['variance']
df_spy_var = df_spy_var.rename_axis('eom')


#=================================
#     Save combined results
#=================================

(monthly_returns
 .merge(df_spy_var, on = ['eom'], how = 'left')
 .to_sql(name = "SPY", con = Benchmarks, if_exists = "replace", index = False)
 )

Benchmarks.close()