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

#### 3. Macro Signal Processing Functions

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

#### 4. Market Cap & Weight Calculation Functions

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

#### 5. Covariance & Risk Calculation Functions

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

#### 6. Black-Litterman Framework Functions

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
  - Ensures inputs are properly shaped numpy arrays
- **Returns**: Tuple of (posterior_mean_returns, posterior_covariance_matrix)

---

#### 7. Portfolio Optimization Functions

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

#### 8. Portfolio Performance & Return Functions

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

##### `tracking_error_and_information_ratio(portfolios_dict, benchmark_weights, sigma_stocks, portfolio_returns_annualized, benchmark_return_annualized)`
- **Purpose**: Calculate tracking error and information ratio for multiple portfolios vs benchmark
- **Key Features**:
  - **Tracking Error**: √(w_active' × Σ × w_active) where w_active = portfolio_weights - benchmark_weights
  - **Information Ratio**: (Portfolio Return - Benchmark Return) / Tracking Error
  - Handles division by zero (sets IR to NaN if tracking error = 0)
  - All values annualized and converted to percentages for display
- **Returns**: DataFrame with columns: Tracking Error (Annualized %), Portfolio Return (Annualized %), Benchmark Return (Annualized %), Excess Return (Annualized %), Information Ratio

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

Macro Processing
├── process_macro_signals()
└── generate_tactical_views()

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

