### Functions Overview:

This document provides a comprehensive summary of all functions defined in the project, with special attention to key implementation details and edge cases.

---

#### 1. Data Processing & Analysis Functions

##### `factor_return_metrics(returns_df, start_date, end_date)`
- **Purpose**: Calculate financial performance metrics for factors or portfolios over a specified period
- **Key Features**:
  - Calculates annualized mean return (×252) and volatility (×√252)
  - Computes Sharpe ratio with division-by-zero protection
  - Calculates maximum drawdown using cumulative wealth tracking
  - Returns DataFrame with metrics as rows, assets/factors as columns
- **Returns**: DataFrame with columns: Annualized Mean, Annualized Volatility, Sharpe Ratio, Max Drawdown

##### `get_famafrench_daily(name, start=None, end=None)`
- **Purpose**: Download and parse Ken French daily factor datasets directly from the Dartmouth FTP site
- **Key Features**:
  - Fetches zipped CSV files from Ken French's data library (e.g. momentum, FF research factors)
  - Parses the raw file format manually to avoid `pandas_datareader` date-parsing failures on trailing commas
  - Strips phantom trailing fields from each line before building a clean CSV string
  - Detects data rows via 8-digit date pattern and reconstructs column headers automatically
  - Optional date filtering via `start` and `end` parameters
- **Returns**: DataFrame of daily factor returns with a datetime index

##### `split_data(data, train_start_date, train_end_date, test_start_date, test_end_date)`
- **Purpose**: Split time series data into training and test sets
- **Key Features**:
  - Simple date-based slicing using pandas `.loc[]`
  - Ensures no overlap between train and test periods
- **Returns**: Tuple of (train_data, test_data) DataFrames

---

#### 2. Regression Analysis Functions

##### `regression_analysis(Y_data, X_data)`
- **Purpose**: Generic OLS regression function for both asset and factor regressions
- **Key Features**:
  - Handles both Series and DataFrame inputs (auto-converts to DataFrame)
  - Adds constant term for intercept automatically
  - Finds common indices between Y and X data
  - Extracts Alpha, Betas, t-statistics, R-squared, Adjusted R-squared, F-statistic
- **Returns**: DataFrame with regression results (one row per dependent variable)

##### `univariate_rolling_regression(Y_data, X_data, window=252)`
- **Purpose**: Perform rolling window OLS regression for univariate analysis
- **Key Features**:
  - Handles both Series and DataFrame inputs
  - Default window: 252 trading days (1 year)
  - For each window, extracts: Alpha, Beta, R-squared
  - Handles edge case: window larger than available data (returns empty DataFrame with warning)
  - Handles both 1D and 2D X_data arrays
- **Returns**: DataFrame indexed by Date with columns: Factor, Alpha, Beta, R-squared

##### `multivariate_rolling_regression(Y_data, X_data, window=252)`
- **Purpose**: Perform rolling window OLS regression for multivariate factor models
- **Key Features**:
  - Designed for portfolio returns (Y) regressed against multiple factors (X)
  - Default window: 252 trading days
  - Extracts Alpha and Beta for each factor (Beta_MKT_RF, Beta_SMB, Beta_HML, Beta_MOM)
  - Returns dictionary where keys are portfolio names, values are DataFrames
  - Handles missing factors gracefully (sets Beta to NaN)
- **Returns**: Dictionary {portfolio_name: DataFrame} with Date index and columns: Portfolio, Alpha, Beta_{factor}, R-squared

---

#### 3. Macro Signal Processing Functions — LEGACY, superseded

> These two functions implement the original view pipeline: daily macro Z-scores
> fed through a linear regression to produce Q. That specification returned
> R² of 0.001–0.005 with no coefficient near significance, and has been replaced
> by the regime-conditional means in Section 4. Both are retained because the
> notebook still runs them as documented diagnostics (output is named
> `tactical_views_legacy` so it cannot overwrite the live `tactical_views`).
> They are not part of the Black-Litterman path.

##### `process_macro_signals(macro_df, window=252)`
- **Purpose**: Transform raw macroeconomic data into normalized tactical allocation signals
- **Key Features**:
  - **Yield Curve Slope**: Directly uses T10Y2Y (10Y-2Y Treasury spread)
  - **CPI Inflation**: Converts monthly CPI to year-over-year percentage change, then forward-fills to daily frequency
  - **VIX**: Directly uses VIXCLS (VIX closing price)
  - Normalizes all signals using rolling Z-scores: (value - rolling_mean) / rolling_std
  - Default normalization window: 252 days
  - Returns both raw signals and normalized versions
- **Returns**: DataFrame with both raw signals (Yield_Curve_Slope, CPI_YoY, VIX) and normalized versions (_norm suffix)

##### `generate_tactical_views(macro_signals_df, tactical_results, factors_list, consensus_window=63)`
- **Purpose**: Translate current macroeconomic Z-scores into annualized expected factor returns (Q vector for Black-Litterman)
- **Key Features**:
  - Uses consensus approach: averages last 63 days of macro signals to represent "current state"
  - For each factor:
    - Starts with Alpha (intercept) from tactical regression
    - Adds Beta × Current_Z_Score for each macro variable
    - Annualizes expected return (daily × 252)
    - Calculates signal strength: |expected_return - alpha| / |alpha|
- **Returns**: DataFrame with columns: Expected_Return_Ann, Alpha_Ann, R_squared, Signal_Strength

---

#### 4. Regime Classification & Conditional Statistics

This is the **current** view-generation pipeline. It replaces the linear
specification in Section 3. Operates on the monthly 1983–2024 panel; every
threshold is either a fixed economic constant or an expanding-window statistic,
so no classification uses information unavailable at the time.

##### `label_regimes(df)`
- **Purpose**: Classify each month into discrete macro regime labels, then combine them into the pre-committed conditioning grids
- **Key Features**:
  - **Conditioners** (the three axes that are allowed to feed views):
    - `growth`: `below_trend` if `cfnai` < 0.0, else `above_trend` — CFNAI is constructed so 0 = trend growth
    - `infl_lvl`: `high` if `cpi_yoy` > `cpi_med`, else `low` — compared against its own **expanding median**, not a fixed rate, so "high" means high relative to what had been normal *at that date*. Real-time by construction
    - `policy`: `accommodative` if `real_ff` < 0.0, else `restrictive` — a positive real fed funds rate is restrictive
  - **Grids** (pre-committed factor→conditioner assignment, not re-optimized):
    - `grid_A` = growth × inflation → **HML**
    - `grid_B` = growth × policy → **SMB**
    - `grid_M` = growth alone → **MOM** (marginal; a second axis is not economically motivated)
  - **Diagnostic only — never conditioners**: `curve_d` (inverted if `curve` < 0), `vol_d` (high if `vix` > 20), `dir_d` (rising if `cpi_dir` > 0). `cpi_dir` in particular switches 103 times in 413 months, far too unstable to condition on
  - Expects `df` to contain: `cfnai`, `cpi_yoy`, `cpi_med`, `real_ff`, `curve`, `vix`, `cpi_dir`, already lagged to publication schedule
  - Purely vectorized via `np.where` — no rolling windows or lookback
- **Returns**: DataFrame (same index as `df`) with columns: growth, infl_lvl, policy, grid_A, grid_B, grid_M, curve_d, vol_d, dir_d

##### `episode_id(mask)`
- **Purpose**: Label each contiguous run of `True` with a distinct integer id, so returns can be grouped *within* an episode
- **Key Features**:
  - `(m.diff() == 1).cumsum()` increments the id at each False→True transition
  - `.where(mask)` blanks the id outside the regime, so the result can be passed straight to `groupby`
  - This is the mechanism behind episode clustering: a regime lasting eighteen months supplies one observation, not eighteen
- **Returns**: Series aligned to `mask` — run id where `mask` is True, NaN elsewhere

##### `count_episodes(mask)`
- **Purpose**: Count the number of contiguous `True` runs ("episodes") in a boolean Series — how persistent a regime label is (long stretches vs. frequent short blips)
- **Key Features**:
  - Casts the mask to int and takes `.diff()`: a value of `1` marks a False→True transition, i.e. the start of a new episode
  - **Edge case**: `.diff()` cannot detect a transition at row 0 (no prior row to compare against), so a run already in progress at the first observation is missed — handled explicitly by checking `m.iloc[0] == 1` and adding 1. The `len(m)` guard makes an empty mask return 0 rather than raising
- **Returns**: int — number of distinct contiguous `True` runs

##### `regime_snapshot(period, regimes, macro, spec, sample=None)`
- **Purpose**: Show the classification at a single date together with its distance to each threshold — how close the call was to flipping
- **Key Features**:
  - `spec` maps regime column → `(macro column, threshold, label_if_above)`. The threshold may be a constant **or a column name** (e.g. the expanding median), which is what lets a time-varying threshold be documented the same way as a fixed one
  - Reports margin in the variable's own units **and** scaled by that variable's standard deviation — CFNAI is an index and inflation a decimal rate, so raw margins are not comparable across rows
  - Flags `fragile = True` when the margin is within 0.25 sd of the threshold, i.e. the label could flip on an ordinary data revision
- **Returns**: DataFrame indexed by regime variable, columns: input, value, threshold, margin, margin_/_sd, label, fragile

##### `occupancy(regimes, cols, n_total=None)`
- **Purpose**: Report how much data actually sits behind each regime state, in months *and* in independent episodes
- **Key Features**:
  - For each state: month count, share of sample, episode count (via `count_episodes`), and average episode length
  - `se_inflation` = √(avg episode length) — approximately the factor by which a month-count standard error understates true uncertainty. This is the number that motivates building Ω from episode-clustered errors rather than monthly ones
- **Returns**: DataFrame indexed by (variable, state), columns: months, pct, episodes, avg_ep_len, se_inflation

##### `association(regimes, col_a, col_b, concordant_pairs)`
- **Purpose**: Check whether two regime axes are actually independent, or whether one largely determines the other (which would make a 2×2 grid mostly empty on one diagonal)
- **Key Features**:
  - `cramers_v`: 0 under independence, 1 when one axis determines the other; equals |φ| for a 2×2 table
  - `concordance`: share of months in the economically paired states. Pairs are named **explicitly** via `concordant_pairs` rather than read off the matrix diagonal — the diagonal depends on alphabetical sort order and would pair `high` with `accommodative`, the opposite of what is intended
  - `concordance_if_independent` gives the benchmark those pairs would hit by chance, so the raw concordance is interpretable
- **Returns**: Tuple of (crosstab DataFrame, summary Series with n_months, cramers_v, concordance, concordance_if_independent)

##### `era_split(factor, regime_col, split_year, labels, sample, regimes)`
- **Purpose**: Test whether a regime spread survives **within** each era, or only appears when eras are pooled — the latter would mean an era effect is being mistaken for a regime effect
- **Key Features**:
  - Splits the sample at `split_year` and computes annualized conditional means by regime state separately in each half, plus the spread between states
  - Intended to run on the **full** history (`calib_full` / `reg_calib_full`), not the restricted 1983+ calibration window — the point is to inspect the period that was deliberately excluded
  - Uses **marginal** axes rather than the 2×2 cells: splitting four cells across two eras leaves eight groups, and the pre-1983 groups are too thin to read
  - `sample` and `regimes` are explicit required arguments in the module version (the notebook cell defaults them to full-history globals)
- **Returns**: DataFrame indexed by era, columns: n_months, `{state}_n`, `{state}_mean`, spread

##### `test_drift(regimes_calib, regimes_test, view_period, grids)`
- **Purpose**: Measure how much of the out-of-sample window the view-date regime actually persisted through
- **Key Features**:
  - **POST-HOC ONLY.** Views are formed once at `view_period`; this reports what fraction of the test window the resulting static view applied to. It never feeds calibration
  - Also lists the realised state distribution in the test window, so a view held against a regime that flipped early is visible rather than hidden in the aggregate performance
- **Returns**: DataFrame indexed by grid, columns: view_date_cell, months_matching, months_total, pct_matching, realised_states

##### `conditional_stats(factor_returns, regime_labels, ann=12)`
- **Purpose**: Per-regime factor performance with **episode-clustered** standard errors — the table that produces both Q and the proposed Ω
- **Key Features**:
  - Groups returns by regime state, then by contiguous episode within each state
  - `se_naive_ann`: treats months as independent. Reported **only** to show how much precision that assumption manufactures
  - `se_cluster_ann`: std of episode means / √(n_episodes) — the figure that should inform Ω. NaN when a state has only one episode, since a single episode supports no dispersion estimate
  - `se_ratio` = cluster / naive makes the overstatement explicit
  - `ep_min` / `ep_max`: worst and best **cumulative** episode return, so a mean driven by one episode is visible
  - Also reports n_months, n_episodes, avg_ep_len, mean_ann, std_ann, hit_rate
- **Returns**: DataFrame indexed by regime state

---

#### 5. Market Cap & Weight Calculation Functions

##### `weight_calc(stock_list, factor_list, start_date, end_date, test_start_date, test_end_date)`
- **Purpose**: Calculate market-capitalization weights for stocks, handling multiple scenarios for shares outstanding
- **Key Features**:
  - **Critical**: Uses unadjusted prices (`auto_adjust=False`) for accurate market cap calculation
  - **Shares Outstanding Logic - Multiple Scenarios**:
    1. **Split Check**: First checks if stock split occurred in test period (test_start_date to test_end_date)
    2. **Scenario A - Split in Test Period**: 
       - Uses `ticker.info.get("sharesOutstanding")` (current snapshot)
       - Rationale: Historical shares data may be unreliable after splits
    3. **Scenario B - No Split in Test Period**:
       - Attempts to use `ticker.get_shares_full(start_date, end_date)` (historical time series)
       - Falls back to `ticker.info.get("sharesOutstanding")` if historical data unavailable or empty
   - **Error Handling**: Raises ValueError if no shares data available
   - **Factor Handling**: Appends factors to DataFrame with zero market cap and zero weights
   - Calculates weights as: Market Cap / Total Market Cap
- **Returns**: DataFrame with columns: Market Cap, Total Market Cap, Weight (index includes stocks + factors)

---

#### 6. Covariance & Risk Calculation Functions

##### `calculate_covariance_matrices(returns_df, weights_df, annualize=True)`
- **Purpose**: Calculate covariance matrices using both conventional and Ledoit-Wolf shrinkage methods
- **Key Features**:
  - **Conventional Covariance**: Standard sample covariance matrix
  - **Ledoit-Wolf Shrinkage**: Reduces estimation error by shrinking toward structured estimator
  - Maintains asset order from `weights_df` to ensure consistency
  - Drops NaN rows before calculation
  - Annualizes by default (multiplies by 252 trading days)
  - Returns shrinkage factor (0-1): closer to 1 = more shrinkage, closer to 0 = less shrinkage
- **Returns**: Dictionary with keys: 'conventional', 'ledoit_wolf', 'shrinkage_factor'

##### `lambda_calc(market_returns_df)`
- **Purpose**: Calculate risk aversion parameters (λ) for different investor types
- **Key Features**:
  - **Market Lambda**: Annualized mean / Annualized variance
  - **Trustee Lambda**: 2 × Market Lambda (more risk-averse)
  - **Kelly Lambda**: Fixed at 1.0 (aggressive risk-taking)
- **Returns**: Dictionary with keys: 'lambda_market', 'lambda_trustee', 'lambda_kelly'

---

#### 7. Black-Litterman Framework Functions

##### `generate_P_matrix(sigma_df, factor_list)`
- **Purpose**: Generate the P (pick) matrix that maps tactical views to factors in Black-Litterman
- **Key Features**:
  - Creates matrix with zeros for all stock positions
  - Sets ones for corresponding factor positions
  - Each row represents one view, each column represents one asset
  - Dimensions: (num_views × num_assets)
- **Returns**: NumPy array (P matrix)

##### `meucci_black_litterman(mu_prior, sigma_prior, P, Q, omega, tau)`
- **Purpose**: Calculate posterior returns and covariance using Meucci's Black-Litterman formulation
- **Key Features**:
  - **Inputs**: Prior returns (μ), prior covariance (Σ), P matrix, Q vector (views), Omega (view confidence), tau (scaling)
  - **Posterior Mean (μ_BL)**: 
    - Formula: μ_prior + (τ·Σ)·P'·(P·(τ·Σ)·P' + Ω)⁻¹·(Q - P·μ_prior)
    - Adjusts prior based on view differences weighted by confidence
  - **Posterior Covariance (Σ_BL)**:
    - Formula: (1+τ)·Σ - (τ²·Σ)·P'·(P·(τ·Σ)·P' + Ω)⁻¹·P·Σ
    - Reflects reduced uncertainty after incorporating views
  - **Omega shape**: accepts a 1-D vector of per-view variances (diagonalised internally via `np.diag`) or a full k×k matrix. The internal diagonalisation guards against a subtle bug: adding a 1-D vector to P·(τ·Σ)·P' broadcasts it across rows, producing a near-singular matrix whose inverse — and therefore the entire posterior adjustment — is numerical noise
  - Ensures inputs are properly shaped numpy arrays
- **Returns**: Tuple of (posterior_mean_returns, posterior_covariance_matrix)

---

#### 8. Portfolio Optimization Functions

##### `optimize_mean_variance(mu, sigma, lam, asset_names, factor_names, factor_weight_lower=-1, factor_weight_upper=1)`
- **Purpose**: Optimize portfolio weights using mean-variance optimization (maximize utility)
- **Key Features**:
  - **Objective Function**: Maximize U = w'μ - (λ/2)·w'Σw (minimize negative utility)
  - **Constraints**:
    - Stock weights sum to 1 (fully invested)
    - Factor overlay weights sum to 0 (no net factor exposure)
  - **Bounds**:
    - Stocks: (0, 1) - no short selling
    - Factors: (factor_weight_lower, factor_weight_upper) - default (-1, 1)
  - **Initial Guess**: Equal-weight stocks (1/n for each stock)
  - Uses SLSQP optimization method
- **Returns**: NumPy array of optimized weights (or None if optimization fails)

##### `optimize_max_sharpe(mu, sigma, asset_names, factor_names, factor_weight_lower=-1, factor_weight_upper=1, rf=0)`
- **Purpose**: Optimize portfolio weights to maximize Sharpe Ratio
- **Key Features**:
  - **Objective Function**: Maximize (w'μ - rf) / √(w'Σw) (minimize negative Sharpe)
  - **Same Constraints & Bounds** as mean-variance optimization
  - Handles division by zero in volatility calculation (uses max(portfolio_var, 1e-10))
  - Default risk-free rate: 0 (for excess returns)
  - Uses SLSQP optimization method
- **Returns**: NumPy array of optimized weights (or None if optimization fails)

##### `optimize_erc(sigma, stock_names)`
- **Purpose**: Optimize portfolio weights to achieve Equal Risk Contribution (Risk Parity)
- **Key Features**:
  - **Objective Function**: Minimize variance of Risk Contributions
    - Calculates Marginal Contribution to Risk (MCR) = (Σ·w) / portfolio_vol
    - Calculates Risk Contribution (RC) = w × MCR
    - Minimizes sum of squared differences between all RC pairs
  - **Constraint**: Weights sum to 1
  - **Bounds**: (0, 1) for all assets - long-only
  - **Initial Guess**: Equal weight (1/n)
  - Uses SLSQP with tolerance 1e-10
  - **Note**: Typically performed on stocks only (factors excluded)
- **Returns**: Pandas Series of optimized weights (or None if optimization fails)

---

#### 9. Portfolio Performance & Return Functions

##### `portfolio_return_metrics(portfolio_name, returns_df, weights, start_date, end_date, is_market_return=False)`
- **Purpose**: Calculate comprehensive financial metrics for a single portfolio
- **Key Features**:
  - **Metrics Calculated**:
    - Annualized Mean Return (×252)
    - Annualized Volatility (×√252)
    - Annualized Geometric Mean Return
    - Sharpe Ratio (with division-by-zero protection)
    - Maximum Drawdown (using cumulative wealth)
    - Cumulative Return
    - Diversification Ratio (weighted avg vol / portfolio vol) - only for non-market returns
  - **Special Handling**: 
    - If `is_market_return=True`, uses returns directly (no weight calculation)
    - Otherwise, calculates weighted portfolio returns
  - Returns both metrics DataFrame and cumulative return Series
- **Returns**: Tuple of (metrics_DataFrame, cumulative_return_Series)

##### `calculate_portfolio_return_series(returns_df, weights, start_date, end_date, is_market_return=False)`
- **Purpose**: Calculate portfolio return time series (not metrics) for a given period
- **Note**: Internal helper — not called directly by the notebook; it is the engine behind `generate_portfolio_returns_dataframe()`
- **Key Features**:
  - Handles both Series and array inputs for weights (converts arrays to Series)
  - Finds common index between returns and weights
  - Calculates daily portfolio returns: Σ(weights × asset_returns)
  - Returns time series of portfolio returns
- **Returns**: Pandas Series of portfolio returns

##### `generate_portfolio_returns_dataframe(portfolio_weights_dict, returns_df, start_date, end_date, stock_list=None, use_stocks_only=True)`
- **Purpose**: Generate DataFrame of portfolio return series for multiple portfolios
- **Key Features**:
  - **Input Flexibility**: Accepts either:
    - Dictionary: {portfolio_name: weights_Series}
    - DataFrame: columns = portfolios, rows = assets
  - **Stock Filtering**: If `use_stocks_only=True`, filters weights to stock_list only
  - Useful for rolling regression analysis (portfolios as columns, dates as rows)
- **Returns**: DataFrame with dates as index, portfolio names as columns

##### `tracking_error_and_information_ratio(portfolios_dict, benchmark_weights, sigma_stocks, portfolio_returns_annualized, benchmark_return_annualized, returns_df=None, start_date=None, end_date=None, periods_per_year=252)`
- **Purpose**: Active risk and risk-adjusted active return vs. a benchmark, reported **both ex-ante and ex-post**
- **Key Features**:
  - **TE ex-ante**: √(w_active' × Σ × w_active), where `sigma_stocks` is the **training-window** Ledoit-Wolf covariance. A *forecast* of active risk
  - **TE ex-post**: annualized standard deviation of the realised daily active return series, `(returns_df × w_active).sum(axis=1)`, over the test window. What actually happened
  - **`TE Ratio (Post/Ante)`**: the diagnostic. Above 1 means realised active risk exceeded the forecast. Expect this above 1 whenever the weights were optimised on `sigma_stocks` itself — the optimiser exploits estimation error in Σ, so the in-sample forecast is biased low. Largest for unconstrained concentrated portfolios, smallest for ERC
  - **`IR Ex-Post` only**: realised excess return ÷ realised TE. Both terms are measured on the same window and basis
  - **The mixed ratio is deliberately not returned.** Realised excess return ÷ ex-ante TE is neither an ex-ante IR (which needs *expected* active return, `w_active' μ_posterior`, in the numerator) nor an ex-post one. The notebook prints it separately, labelled superseded, only to size the correction
  - `returns_df` must be on the same basis (excess vs. total) as `portfolio_returns_annualized`; the notebook passes `excess_return_data_test` to match
  - Omitting `returns_df` returns the ex-post columns as NaN rather than silently falling back to the mixed figure
  - Raises `ValueError` if portfolio and benchmark weights do not jointly cover `sigma_stocks.index` (pandas index alignment would otherwise produce a NaN tracking error silently), or if `returns_df` lacks a required ticker column
  - Division-by-zero protection: IR and TE ratio are NaN when the relevant tracking error is 0
- **Returns**: DataFrame indexed by portfolio name, columns: TE Ex-Ante (Annualized %), TE Ex-Post (Annualized %), TE Ratio (Post/Ante), Portfolio Return (Annualized %), Benchmark Return (Annualized %), Excess Return (Annualized %), IR Ex-Post

---




## Key Implementation Notes

### Shares Outstanding Handling (Critical)
The `weight_calc()` function implements a sophisticated fallback mechanism for shares outstanding:

1. **Primary Check**: Detects stock splits in test period
2. **Split Scenario**: Uses current snapshot (`ticker.info`) to avoid historical data issues
3. **No Split Scenario**: Attempts historical time series, falls back to snapshot if unavailable
4. **Rationale**: Stock splits can make historical shares data unreliable; current snapshot is more accurate post-split. Yahoo Finance adjusts closing price, even after pulling auto adjust = False data. 

### Factor Treatment
- Factors are treated as **latent return drivers** (not direct capital allocations)
- Factor weights always sum to zero in optimization constraints
- Factors influence stock selection through correlation but don't consume portfolio capital
- This aligns with institutional practice (factors accessed via ETFs/derivatives, not direct holdings)

### Optimization Constraints
All optimization functions enforce:
- **Stock weights sum to 1**: Fully invested portfolio
- **Factor weights sum to 0**: No net factor exposure (factors are overlays)
- **Stock bounds**: (0, 1) - no short selling
- **Factor bounds**: Configurable (default: -1 to 1)

### Error Handling
- All functions include validation for empty data
- Division-by-zero protection in Sharpe ratio calculations
- Window size validation in rolling regressions
- Graceful fallbacks in shares outstanding retrieval

---

## Function Dependencies

```
Data Processing
├── get_famafrench_daily()
├── split_data()
└── factor_return_metrics()

Regression Analysis
├── regression_analysis()
├── univariate_rolling_regression()
└── multivariate_rolling_regression()

Macro Processing (LEGACY - superseded, retained as documentation)
├── process_macro_signals()
└── generate_tactical_views()

Regime Classification & Conditional Statistics (current view pipeline)
├── label_regimes()
├── episode_id()
├── count_episodes()
├── regime_snapshot()
├── occupancy()          -> count_episodes()
├── association()
├── era_split()
├── test_drift()
└── conditional_stats()  -> episode_id()

Portfolio Construction
├── weight_calc()
├── calculate_covariance_matrices()
└── lambda_calc()

Black-Litterman
├── generate_P_matrix()
└── meucci_black_litterman()

Optimization
├── optimize_mean_variance()
├── optimize_max_sharpe()
└── optimize_erc()

Performance Analysis
├── portfolio_return_metrics()
├── calculate_portfolio_return_series()
├── generate_portfolio_returns_dataframe()
└── tracking_error_and_information_ratio()


```

