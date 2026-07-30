# Black-Litterman Portfolio Construction with Regime-Conditional Factor Views

This project builds a Meucci-style Black-Litterman allocation in which the factor views are
regime-conditional: the expected return of each factor is its historical mean *given the current
macroeconomic regime*, with view uncertainty measured from the data rather than assumed.

## Project Objective

The project aims to bridge passive equilibrium allocation and active tactical allocation by:

- starting from market-implied equilibrium returns (Black-Litterman prior),
- classifying the macro regime in real time from published data,
- setting each factor view (SMB, HML, MOM) to its regime-conditional historical premium,
- sizing view confidence from the estimation error of that premium,
- and evaluating how the posterior changes portfolio weights and out-of-sample behavior.

In practical terms: the goal is to test how much a transparent, pre-committed macro-regime signal
actually moves a portfolio once its uncertainty is stated honestly — not to manufacture alpha.

## Methodology Overview

### 1) Data and Universe Design

- 13 large-cap stocks chosen ex-ante for sector diversity and heterogeneous macro sensitivity.
  (Two early screening filters — pairwise correlation and factor-regression R² — were removed
  after diagnostic review showed they selected on market beta, not factor exposure. Both
  regressions are retained in the notebook as reported diagnostics.)
- Daily data with a clean train/test split for covariance estimation and evaluation; a separate
  monthly panel (1983–2024) for view calibration.
- Every macro input is lagged to its real publication schedule. Every classification threshold is
  a fixed economic constant or an expanding-window statistic — no full-sample information leaks
  into any regime label.

### 2) Regime Classification and View Generation

- Three real-time regime axes, each with an independently justified threshold:
  - **Growth**: CFNAI 3-month MA vs 0 (the index is constructed so 0 = trend growth),
  - **Inflation**: CPI YoY vs its own expanding median ("high relative to what had been normal"),
  - **Policy**: real fed funds rate vs 0 (positive = restrictive).
- Pre-committed factor→grid assignment (documented before results were computed):
  HML on growth × inflation, SMB on growth × policy, MOM on growth alone.
- The `Q` vector is the conditional mean of each factor in the regime cell occupied at the view
  date. The **regime tilt** (conditional minus unconditional mean) is reported alongside it,
  because that difference — not `Q` itself — is what the regime machinery contributes.
- **Uncertainty is clustered by episode, not by month**: a regime lasting eighteen months is one
  observation, not eighteen. The episode-clustered standard error of each conditional mean is
  reported next to the naive one, and is typically several times larger.
- An earlier view pipeline (daily macro Z-scores through a linear regression on yield curve, CPI
  and VIX) is retained in the notebook, marked superseded, with the diagnosis of why it failed
  (persistence collapses the effective sample size; levels of published macro data shouldn't
  predict returns; constant slopes contradict the regime premise).

### 3) Black-Litterman Integration (Meucci formulation)

- Stock prior from reverse optimization on market-cap weights; factor prior set to the
  unconditional mean over the *same* window as the views, so the view innovation
  `Q − Pπ` is exactly the regime tilt.
- **Ω = diag(SE²_cluster)** — view confidence is the measured estimation error of each view,
  applied as a diagonal matrix. This removes the τ-dependence of the common
  `Ω = diag(PτΣP′)` calibration and lets sparse regimes self-penalise.
- τ = 0.025, a stated prior-confidence constant (1/T does not apply to a reverse-optimized prior
  against an annualized Σ; the reasoning is documented in the notebook).
- The P matrix selects factors directly, so views reach stocks through the stock–factor
  covariance blocks of the Ledoit-Wolf Σ (univariate betas), not through regression loadings.

### 4) Portfolio Construction and Evaluation

Optimization styles compared, all on the same posterior:

- Mean-Variance at three risk-aversion levels (Kelly λ=1, market λ, trustee 2× market λ)
- Maximum Sharpe
- Equal Risk Contribution (ERC), plus equal-weight and naive risk-parity baselines

Evaluation covers out-of-sample performance, active positioning, tracking error (reported both
**ex-ante** from the training covariance and **ex-post** from realized active returns, with the
information ratio computed only against the ex-post figure), factor-exposure regressions, rolling
regressions, and Euler-verified risk contributions.

## Key Findings

- **With honestly-sized uncertainty, the views barely move the posterior.** The current regime's
  tilts are a fraction of a percent on SMB and HML and a two-to-three point markdown on momentum;
  the posterior sits within a fraction of a percentage point of the prior. That is the framework
  correctly reporting that ~40 years of regime history do not justify large departures from
  equilibrium.
- **The optimization objective, not the views, drives portfolio composition.** The same posterior
  produced everything from a two-stock portfolio (Kelly) to an eleven-stock defensive book
  (trustee risk aversion) to full 13-stock risk parity (ERC).
- **In the single test year, return-seeking optimizers beat the market and risk-balanced
  constructions did not** — but factor regressions attribute the outperformance to market beta
  and mega-cap concentration, with no statistically significant alpha anywhere, and one-year
  information ratios carry standard errors larger than every observed value.
- **Conservative optimization was the all-round winner**: high risk aversion delivered the best
  realized Sharpe with below-market volatility and drawdown. Aggressive concentration (Kelly)
  delivered the worst risk-adjusted result — its volatility drag consumed most of its return.
- **ERC provided the best drawdown control and diversification**, at the cost of below-market
  return in a mega-cap-led year. Being return-agnostic, it is also a useful control: its results
  are invariant to everything the views do.
- Results are best treated as **illustrative evidence of framework behavior**, not proof of
  persistent alpha.

## Conclusion

The notebook shows that regime-conditional factor views can be integrated into Black-Litterman in
a coherent, transparent way, with view uncertainty measured rather than assumed. The economic
value of such tilts is modest — and stating that honestly is the point of the exercise. Deployment
would additionally require:

- validation across multiple market environments (rolling / regime-robust backtesting),
- sensitivity analysis to alternative regime definitions,
- reconciliation of univariate vs multivariate factor loadings (which names a momentum view
  actually tilts toward),
- and realistic implementation constraints (turnover, costs, market frictions).

## Repository Contents

| File | Role |
|---|---|
| `Portfolio Construction Black Litterman — Code.ipynb` | Main notebook: data, regimes, views, BL, optimizers, performance |
| `Portfolio Construction Black Litterman — Theory.ipynb` | Conceptual and mathematical background (no code, no result-specific numbers) |
| `bl_functions.py` | Shared function library, imported by the code notebook |
| `FUNCTIONS.MD` | Documentation for every function in `bl_functions.py` |
| `monthly_factors_macro.csv` | Cached monthly factor + macro panel (pinned for reproducibility) |

## Reproducibility Notes

- Environment: Python / Jupyter
- Main stack: `pandas`, `numpy`, `statsmodels`, `scipy`, `scikit-learn` (Ledoit-Wolf),
  `yfinance`, `pandas-datareader`, `plotly`
- The monthly factor/macro panel is cached to CSV so the view pipeline is reproducible offline;
  daily stock/factor data are pulled from Yahoo Finance, FRED and Ken French's data library at
  run time. Market-cap weights use current shares outstanding as a snapshot — pinning them to a
  committed CSV is a known open item.
- Keep the train/test separation intact when rerunning to preserve the validity of the
  out-of-sample evaluation.
