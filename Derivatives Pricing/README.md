
## Objective

The objective for this project is pricing and analysis of:

- European call options
- Binary (cash-or-nothing) call options

using the risk-neutral valuation framework:

\[
V(S,t)=e^{-r(T-t)}\mathbb{E}^Q[\text{Payoff}(S_T)]
\]

The implementation compares three stock-path simulation approaches:

- Euler-Maruyama
- Milstein
- Exact GBM closed-form transition

## What Was Implemented

The notebook delivers a full report-style workflow including:

- model setup and assumptions under GBM
- Black-Scholes analytical pricing benchmarks (European and binary)
- Monte Carlo pricing functions with standard error estimates
- convergence diagnostics:
  - strong convergence (pathwise RMSE vs exact GBM)
  - weak convergence (pricing error vs benchmark)
- Monte Carlo sampling study (effect of number of paths on error/runtime)
- sensitivity analysis for:
  - volatility
  - strike
  - time to maturity
  - risk-free rate

## Key Findings

- Milstein showed superior **strong convergence** behavior relative to Euler-Maruyama, consistent with theory.
- All three methods produced option prices close to Black-Scholes benchmarks for this GBM setting.
- Sampling error can dominate discretization error at finite path counts, reinforcing the need for adequate Monte Carlo sample sizes.
- Runtime/accuracy trade-off is method-dependent:
  - Euler-Maruyama is computationally efficient and simple.
  - Milstein generally improves pathwise accuracy with moderate extra cost.
  - Exact GBM is a benchmark under GBM dynamics but still has computational overhead in simulation contexts.


## Reproducibility Notes

- Language: Python (Jupyter Notebook format as required)
- Typical libraries used: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `time`
- Results are generated via Monte Carlo simulation and may vary slightly with random sampling unless a fixed seed is used.

