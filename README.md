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

## Portfolio construction (how return forecasts become an investable portfolio)

I implement a **predict-then-optimise** portfolio construction pipeline that maps **1-month ahead ML return forecasts** into **implementable, benchmark-like, long-only** S&P 500 portfolios using **transaction-cost-aware optimisation** with **quadratic price impact (Kyle’s lambda)**.

### Optimisation problem (myopic, cost-aware, benchmarked risk)

For each month, the optimiser chooses **long-only weights that sum to 1** on the active S&P 500 constituents. Thus, at each beginning of the month, the following objective is solved
Thus, at the beginning of each month $t-$, the manager solves:

At the beginning of each month \( t \), the portfolio weights \( \pi_t \) are chosen by solving:

$$
\max_{\pi_t}
\; \hat{r}_t^\top \pi_t
- w_t (\pi_t - \bar{\pi}_t)^\top \Lambda_t (\pi_t - \bar{\pi}_t)
$$

where

- \( \hat{r}_t = \mathbb{E}[r_t \mid \mathcal{F}_{t-1}] \) are 1-month ahead return forecasts  
- \( \bar{\pi}_t = G_{t-1}\pi_{t-1} \) are the drifted (no-trade) portfolio weights  
- \( \Lambda_t \) is the diagonal matrix of Kyle’s lambda (price impact)  
- \( w_t \) is assets under management (AUM), scaling implementation costs  

subject to

$$
0 \le \pi_t \le \pi_{\max}
$$

$$
\mathbf{1}^\top \pi_t = 1
$$

$$
\pi_t^\top \Sigma_t \pi_t \le \sigma_{B,t}^2
$$

where \( \Sigma_t = \mathbb{E}[\Sigma_t \mid \mathcal{F}_{t-1}] \) is the ex-ante covariance matrix and  
\( \sigma_{B,t}^2 \) is the EWMA-estimated variance of the S&P 500 benchmark.

In the implementation, the optimisation uses **softmax** to enforce the simplex constraint ($\bm{1}^\top\bm{\pi}_{t-}=1$, $\bm{\pi}_{t-}\ge 0$) and applies **ReLU penalty terms** for the remaining inequality constraints (concentration limits / variance cap). Transaction costs are modelled as **quadratic price impact (Kyle’s $\lambda$)** on deviations from the **drifted no-trade baseline** $\bm{G}_{t-1}\bm{\pi}_{t-1}$, scaled by AUM $w_{t-}$.

**Key features**
- **Kyle-lambda / quadratic impact / implementation shortfall**: costs scale with trade size squared
- **Endogenous turnover regularisation**: TC term acts like an **$\ell_2$ (Ridge) penalty on trades** (turnover shrinkage)
- **Benchmarked risk control**: **volatility targeting / tracking-risk style constraint** via $\sigma_t^{B}$ (EWMA-estimated benchmark vol)
- Optional **single-name constraints / concentration limits** ($\pi_{\max}$)
- Optional **benchmark-relative weight bounds** from market equity (`me`) or a **flat max-weight cap**

This is a **myopic** (single-period) policy that is computationally robust and easy to stress-test across transaction-cost regimes.

---

### Numerical solution: differentiable constrained optimisation in PyTorch
The optimisation is solved via **gradient ascent** using PyTorch:

- Parameterise weights using **softmax(logits)**:
  - `π = softmax(z)` ensures **long-only** and **fully invested** (`∑ π = 1`) by construction
- Impose inequality constraints using **penalty functions**:
  - max-weight violations: `ReLU(π − π_max)` (optional)
  - min-weight violations: `ReLU(π_min − π)` (optional)
  - variance violation: `ReLU(π' Σ π − max_var)`
- Gradient ascent via **Adam**, learning rate of 0.01.
- 
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
