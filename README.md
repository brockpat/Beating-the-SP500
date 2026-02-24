# Beating the S&P 500 — The Virtue of Transaction Costs (VoT)

This repository contains the code for my paper **“Beating the S&P 500: The Virtue of Transaction Costs” (Feb 18, 2026)**.  

I show that **stock return forecasts using machine learning can be translated into an investable portfolio in a liquid large-cap universe** once portfolio construction is done with **transaction-cost-aware optimisation**. The main idea is that transaction costs **regularise** trading, so that the resulting portfolio doesn't overtrade on the noise machine learning forecasts which is responsible for the portfolio's performance.

---

## Overview

### The problem
ML return forecasts often look strong **gross**, but performance frequently collapses **after transaction costs and realistic constraints**, especially in **active S&P 500 constituents**. A key failure mode is **naive forecast→trade mappings** (e.g., ranks/deciles) that convert weak/noisy signals into **excessive turnover**.

### The idea
I implement a modular **predict-then-optimise** pipeline:

1. **Predict** 1-month ahead expected returns using standard ML models (kept separate from portfolio choice):
   - **XGBoost (XGB)**
   - **Transformer (TF)**
   - **Instrumented Principal Component Analysis (IPCA)**
   - **Random Fourier Features (RFF)**

2. **Optimise** portfolio weights:
   - **Long-only, fully invested** in active S&P500 stocks
   - **Volatility benchmarking** (risk controlled to be market-like; MV as robustness)
   - **Quadratic price-impact transaction costs** (Kyle-lambda style; stock-specific liquidity)
   - Optional **single-name concentration limit**

### Evaluation setting (demanding by design)
- Universe: **active S&P500 constituents**
- Frequency: **monthly rebalancing**
- Sample: **2004–2024**
- Primary comparison: net-of-costs performance vs **S&P 500 benchmark** 

### Core mechanism: “Virtue of Transaction Costs” (VoT)
Quadratic price impact makes active rebalancing costly **in proportion to trade size**, so when embedded in the optimiser it acts like an **endogenous Ridge-type penalty on trades**.  
Backtest results from monthly trading from 2004-2024: **optimising as if costs are large improves benchmark-relative performance** by shrinking noisy trades (less overtrading). Hence, acting as if transaction costs are large strictly improves performance. 

---

## Repository File Overview

- `General_Functions.py` — Shared modules
- `Data_Preprocessing.py` — Builds the core monthly signals dataset from raw inputs (e.g., CRSP/market/risk-free data)
- `Feature_Engineering.py` — Standardises signals cross-sectionally (z-scores/ranks) and prepares model-ready feature sets.
- `SP500_Constituents.py` — Constructs the monthly active S&P 500 membership panel used to define the investable universe.
- `SPY_return.py` — Downloads/constructs SPY benchmark return series for benchmarking and volatility targeting.
- `Estimate Covariance Matrix.py` — Estimates stock return covariance/correlation structures for portfolio optimisation inputs.
- `XGBoost.py` — Stock return forecasts using XGBoost.
- `Transformer.py` — Stock return forecasts using a Transformer.
- `IPCA.py` — Stock return forecasts using IPCA.
- `RFF.py` — Stock return forecasts using Random Fourier Feature Ridge Regression.
- `Portfolio_Optimiser.py` — Converts machine learning return forecasts into constrained long-only portfolios with volatility controls and transaction-cost-aware optimisation.
- `Results.py` — Aggregates backtests and computes/report performance analytics (Sharpe/IR/drawdowns/turnover, comparisons, and robustness outputs).
