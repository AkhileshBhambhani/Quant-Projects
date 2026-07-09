# SVM-Based Directional Signal for SPY

A methodology-first exploration of using Support Vector Machines to predict short-horizon directional moves in SPY (S&P 500 ETF), built around leakage-free time-series cross-validation, stationary feature engineering, and honest out-of-sample evaluation.

## What this is

This notebook predicts whether SPY's next-day close will exceed today's close by more than 0.25%, using a pipeline of engineered technical/statistical features, L1-regularized feature selection, and a tuned SVM classifier.


## Pipeline overview

1. **Stationary feature engineering** — every feature is a ratio or return, not a raw price level (e.g. `close/SMA - 1` instead of raw `SMA`). This matters specifically because SVM is a distance-based kernel method: a non-stationary, trending raw price feature distorts kernel-distance similarity between training and out-of-sample data, independent of any scaler choice.
2. **Chronological train/test split** — no shuffling, to avoid look-ahead bias in the split itself.
3. **Two-stage feature selection**, both fit on the training set only:
   - Correlation-based redundancy filtering (drop one side of any pair correlated above 0.9)
   - L1-regularized (Lasso) logistic regression, tuned via `GridSearchCV` + `TimeSeriesSplit`
4. **Outlier-aware dual scaling** — features are bucketed by IQR-flagged outlier rate and scaled with `RobustScaler` or `StandardScaler` accordingly, with bucket assignment computed strictly from training data.
5. **SVM hyperparameter tuning** — exhaustive `GridSearchCV` over kernel-conditional parameter grids (`rbf`, `linear`, `poly`), scored on F0.5 rather than F1, reflecting the asymmetric cost of false positives (a bad trade) vs. false negatives (a missed opportunity) in a long-only signal strategy.
6. **Out-of-sample-only backtesting** via `pyfolio`, comparing the strategy against buy-and-hold on data the model never trained on.

## Key methodological decisions (and why)

- **Why ratios, not price levels:** raw price-level features (moving averages, bands) drift with the underlying trend. A scaler fit on one price regime doesn't generalize to a different one — normalizing against a contemporaneous reference price fixes this at the feature level rather than papering over it downstream.
- **Why `TimeSeriesSplit` everywhere:** every cross-validated step (correlation filtering excluded, since it's a single point-in-time fit) uses forward-chaining folds, so validation scores reflect genuine walk-forward generalization rather than shuffled-in future information.
- **Why no `SequentialFeatureSelector`:** with only a handful of features surviving Lasso selection, a greedy wrapper search adds meaningful compute for limited expected gain — and by default it uses a non-time-aware internal CV unless `cv=` is explicitly passed, which would quietly reintroduce leakage inside the feature-selection step.
- **Why F0.5 over F1:** false positives (entering a losing trade) carry a real, realized cost; false negatives (missing an up-day) cost nothing directly. F0.5 weights precision roughly twice as heavily as recall to reflect that asymmetry.
- **Why the backtest is restricted to the test set only:** an earlier draft generated trading signals across the full dataset (train + test) before backtesting — this let the model "trade" on data it had already been fit on, inflating apparent performance. `pyfolio`'s `live_start_date` argument only colors a return series by date; it has no awareness of a model's training boundary, so this kind of leakage is easy to introduce without noticing.

## Limitations

- The out-of-sample test period is a single, relatively short contiguous window — risk-adjusted metrics like Sharpe carry wide uncertainty over that length of time.
- One static train/test split tests one regime transition, not many. A rolling/expanding walk-forward backtest (retrain periodically, stitch together multiple honest out-of-sample periods) would be a more rigorous extension.
- Feature selection is staged and greedy (correlation filter → Lasso → scaler bucketing), not jointly optimized with SVM hyperparameters — a deliberate compute/rigor tradeoff, documented rather than hidden.
- This project uses public daily OHLCV data and standard engineered features. No claim is made about tradeable, capacity-scalable alpha.

## Tech stack

`Python`, `scikit-learn` (Pipeline, ColumnTransformer, GridSearchCV, TimeSeriesSplit, SVC, LogisticRegression), `pyfolio`, `pandas`, `numpy`, `plotly`, `matplotlib`, `yfinance`, `quantmod`

## Structure

Single notebook, organized top to bottom: data → feature engineering rationale → train/test split → correlation filtering → outlier profiling & scaling → Lasso feature selection → SVM tuning → diagnostics (confusion matrix, ROC, train/test gap) → strategy vs. benchmark comparison → out-of-sample tear sheet → limitations.