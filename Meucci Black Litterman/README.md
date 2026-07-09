# CQF Final Project - Black-Litterman Portfolio Construction with Factor and Macro Views

This folder contains my CQF Final Project submission, focused on portfolio construction using a Black-Litterman framework augmented with macro-conditioned factor signals.

## Project Objective

The project aims to bridge passive equilibrium allocation and active tactical allocation by:

- starting from market-implied equilibrium returns (Black-Litterman prior),
- generating forward-looking views from macroeconomic signals,
- mapping those views through factor premia (SMB, HML, MOM, with market context),
- and evaluating how posterior views change portfolio weights and out-of-sample behavior.

In practical terms, the goal is to test whether transparent, data-driven macro-factor tilts can improve portfolio construction relative to static allocation choices.

## Methodology Overview

### 1) Data and Universe Design

- Built an equity universe across multiple sectors.
- Applied correlation filtering and factor-explainability checks to retain assets suitable for factor-based modeling.
- Used a clean train/test split to avoid look-ahead bias.

### 2) Factor and Macro Modeling

- Used a four-factor setup (market, size/SMB, value/HML, momentum/MOM).
- Modeled factor expectations using macro indicators:
  - yield curve slope (10Y-2Y),
  - CPI,
  - VIX.
- Generated tactical expected-return views (`Q` vector) from estimated macro-factor relationships.

### 3) Black-Litterman Integration

- Estimated prior equilibrium returns.
- Combined prior and tactical views to obtain posterior expected returns.
- Examined posterior changes across covariance choices and confidence assumptions.

### 4) Portfolio Construction and Evaluation

Compared optimization styles:

- Mean-Variance
- Maximum Sharpe
- Equal Risk Contribution (ERC)

Then evaluated out-of-sample performance, active positioning, tracking-error behavior, factor exposures, rolling regressions, and risk contributions.

## Key Findings

- Utility/optimization choice is a major driver of final portfolio behavior.
- Aggressive return-seeking allocations can improve terminal wealth but increase drawdown risk.
- Risk-balanced constructions (e.g., ERC) provided more stable diversification and downside protection.
- Macro-factor relationships showed limited explanatory power, so tactical alpha should be interpreted cautiously.
- Results are best treated as **illustrative evidence of framework behavior**, not proof of persistent alpha.

## Conclusion

The notebook shows that macro-informed factor views can be integrated into Black-Litterman in a coherent, transparent way. However, economic value depends strongly on:

- stability of macro-factor relationships,
- robustness of view construction,
- investor risk preference,
- and realistic implementation constraints (turnover, costs, market frictions).

Additional multi-regime and rolling-window validation is required before live deployment.

## Folder Contents

- `PC Akhilesh Bhambhani Code.ipynb` - Primary project notebook (methodology, code, outputs, interpretation)
- `PC Akhilesh Bhambhani Code.html` - Rendered notebook output
- `PC Akhilesh Bhambhani REPORT.pdf` - Final report
- `June 25 Final Project Declaration.pdf` - Signed declaration

## Reproducibility Notes

- Environment: Python / Jupyter
- Main stack: `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib` (and related portfolio analytics utilities used in notebook)
- Keep train/test separation intact when rerunning to preserve validity of out-of-sample evaluation.
