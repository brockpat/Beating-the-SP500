# 📈 Beating the S&P 500 — The Virtue of Transaction Costs (VoT)

Machine learning (ML) return forecasts often exhibit strong **gross** backtest performance.  
Yet in liquid large-cap universes such as the S&P 500, these gains frequently disappear **net of transaction costs**.  

This paper shows:

> **ML stock return forecasts remain economically valuable in active S&P 500 stocks — even net of realistic transaction costs — when the portfolio construction is properly disciplined.**

The key contribution is a *forecast-to-trade mapping* that filters out the noise from ML stock return predictions.

Main mechanism:

> Using predicted stock returns, optimising the portfolio **as if transaction costs are large** improves benchmark-relative performance.

Reason:

> Transaction costs enter as a quadratic penalty on active trades. This shrinks forecast-induced reallocations, prevents aggressive rebalancing, and reduces turnover.  
> By penalising noisy trades, transaction costs filter out spurious forecast variation and extract the small but persistent signal in ML return predictions.

In short, I use transaction costs as an economically grounded layer of regularisation on the portfolio construction This solves two problems at once: Return prediction noise is filtered out and transaction costs are accounted for.

---

## 💡 Key Idea

Naive forecast are mapped to trades using ranks/deciles.

Problems:
- Ignore turnover
- Ignore liquidity
- Translate weak cross-sectional predicted return differences into aggressive reallocations
- Ignores transaction costs
- Collapse net of costs

### 🛠️ The solution: Predict–Then–Optimise

A modular two-stage pipeline:

### 1️⃣ Predict (Forecast Returns)

One-month-ahead expected returns. Four machine learning algorithms are considered:

- **XGBoost (XGB)**
- **Transformer (TF)**
- **Instrumented Principal Component Analysis (IPCA)**
- **Random Fourier Features (RFF)**

Forecasting is **fully separated** from portfolio optimisation.

### 2️⃣ Optimise (Cost-Aware Portfolio Construction)

Portfolio weights are chosen subject to:

- Long-only
- Fully invested
- Active S&P 500 constituents only
- Volatility benchmarking (market-like risk)
- Quadratic price-impact transaction costs
- Optional concentration limits

---

## 🎯 Evaluation Setting (Deliberately Demanding)

| Dimension | Setting |
|------------|----------|
| Universe | Active S&P 500 constituents |
| Frequency | Monthly rebalancing |
| Sample | 2004–2024 |
| Benchmark | S&P 500 (SPY total return) |
| Performance | Net of transaction costs |
| Portfolio | Long Only & Fully Invested |

This avoids:

- Microcaps
- Illiquidity effects
- Leverage
- Shorting

---

## 🏆 Main Result

> Optimising **as if transaction costs are large** improves benchmark-relative performance.

### ⚙️ Core Mechanism: The Virtue of Transaction Costs (VoT)

Transaction costs enter the objective as a quadratic penalty term.

- Costs scale with **trade size²**
- Larger AUM ⇒ stronger penalty
- Acts like an **endogenous Ridge-like penalty on active trades**

Large transaction costs:
- Shrink active trades
- Thereby reducing overtrading
- Leading to higher Sharpe and Information ratios

Transaction costs become a **discipline mechanism**, not just an execution drag.

### 📊 Backtest Results (Net of Transaction Costs)

IPCA is the best performing algorithm. All results are fully out of sample.

| Portfolio | μ | σ | Sharpe | TO | IR | MaxD | DCap | α |
|------------|------|------|--------|------|------|--------|------|------|
| **S&P 500** | 0.110 | 0.147 | 0.645 | • | 0.000 | -0.509 | 1.000 | • |
| **IPCA** | 0.142 | 0.158 | 0.802 | 0.054 | 0.512 | -0.473 | 0.973 | 0.031 |

Notes

- **μ**: Annualised mean return  
- **σ**: Annualised volatility  
- **Sharpe**: Annualised Sharpe ratio  
- **IR**: Annualised Information Ratio (vs. S&P 500)  
- **α**: Annualised abnormal return (CAPM alpha)  
- **TO**: Average monthly turnover  
- **MaxD**: Maximum drawdown  
- **DCap**: Drawdown capture ratio

IPCA delivers:

- Higher return than the S&P 500
- Similar volatility  
- Higher Sharpe ratio  
- Strong benchmark-relative performance (IR = 0.512)  
- Low turnover (5.4% per month)  
- High annualised alpha 


![Model Illustration](Fig.svg)

---

# 🧮 Portfolio Construction

## 🧩 Portfolio Choice Problem

Each month, a fund chooses portfolio weights according to:

$$
\begin{aligned}
\max_{\pi_t} \quad
& \hat{r}_{t+1}^\top \pi_t \\
&- w_t (\pi_t - G_{t-1}\pi_{t-1})^\top
\Lambda_t
(\pi_t - G_{t-1}\pi_{t-1})
\end{aligned}
$$

Subject to:

#### Long-Only and (optional) Concentration Limit

$$
0 \le \pi_t \le \pi_{\max}
$$

#### Fully Invested

$$
\mathbf{1}^\top \pi_t = 1
$$

#### Volatility Benchmarking

$$
\sqrt{\pi_t^\top \Sigma_t \pi_t} \le \sigma_t^B
$$


| Symbol | Meaning |
|--------|----------|
| $\pi_t$ | Portfolio weights |
| $\hat{r}_{t+1}$ | Predicted returns |
| $w_t$ | Fund's wealth (AUM) |
| $G_{t-1}$ | Drift adjustment matrix |
| $\Lambda_t$ | Price impact (Kyle’s $\lambda$) |
| $\Sigma_t$ | Covariance matrix of stock returns |
| $\sigma_t^B$ | S&P500 volatility (EWMA estimate) |
| $\pi_{\max}$ | Concentration limit |

## 🔧 Implementation Details

### Constraint Handling

- **Softmax parameterisation**
  - Ensures long-only
  - Ensures fully invested

- **ReLU penalties**
  - Max weight violations
  - Variance constraint violations

### Transaction Costs

- Quadratic impact (Kyle λ)
- Costs applied to deviations from drifted baseline
- Scale with AUM

### Optimisation
Solve the portfolio choice problem numerically.

- PyTorch
- Gradient ascent (Adam)

---

## 🧠 Why This Works

Two layers of regularisation:

1. **ML shrinkage for return predictions** (early stopping / Ridge / tree regularisation)
2. **Transaction cost shrinkage for active trades** 

The second layer reduces aggressive trading based on noisy machine learning return predictions and successfully filters out the (small) predictive signal in these forecasts.

---

# 📂 Repository Structure

### 🗂️ Data & Preprocessing

- `Data_Preprocessing.py`  
  Builds monthly signals dataset

- `Feature_Engineering.py`  
  Cross-sectional standardisation

- `SP500_Constituents.py`  
  Monthly investable universe

- `SPY_return.py`  
  Benchmark return series

- `Estimate Covariance Matrix.py`  
  Factor-based covariance estimation

---

### 🤖 Return Forecasting

- `XGBoost.py`
- `Transformer.py`
- `IPCA.py`
- `RFF.py`

Each produces 1-month-ahead fully out of sample forecasted stock returns.

---

### 📊 Portfolio & Results

- `Portfolio_Optimiser.py`  
  Cost-aware constrained optimisation

- `Results.py`  
  Backtests, performance metrics, turnover, Sharpe, IR, drawdowns

- `General_Functions.py`  
  Shared utilities

---

# 🎓 Conceptual Contribution

This repository demonstrates:

- ML forecasts contain real information in large-cap equities.
- The bottleneck is not prediction.
- The bottleneck is **implementation discipline**.
- Transaction costs provide an economically grounded regularisation device.

---

# 🏁 Bottom Line

Transaction costs are not just friction.

They are a **regularisation tool** that converts ML forecasts into investable alpha in the S&P 500.
