# ============================================================================
# Portfolio Construction Black-Litterman: Function Library
# ============================================================================
# Functions can be imported using from bl_functions import *

import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

__all__ = [
    'factor_return_metrics',
    'get_famafrench_daily',
    'split_data',
    'regression_analysis',
    'univariate_rolling_regression',
    'multivariate_rolling_regression',
    'process_macro_signals',
    'generate_tactical_views',
    'weight_calc',
    'calculate_covariance_matrices',
    'lambda_calc',
    'generate_P_matrix',
    'meucci_black_litterman',
    'optimize_mean_variance',
    'optimize_max_sharpe',
    'optimize_erc',
    'portfolio_return_metrics',
    'calculate_portfolio_return_series',
    'generate_portfolio_returns_dataframe',
    'tracking_error_and_information_ratio',
]


def factor_return_metrics(returns_df, start_date, end_date):
    """
    Calculate financial metrics for selected period

    Parameters:
    -----------
    returns_df (pd.DataFrame): DataFrame containing excess returns
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    --------
    dict: Dictionary containing calculated metrics

    """

    filtered_df = returns_df.loc[start_date:end_date]
    

    # Check if filtered data is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for date range {start_date} to {end_date}")
    
    
    # Calculate Metrics
    metrics = {}

    for col in filtered_df.columns:
        annualized_mean = filtered_df[col].mean() *252
        annualized_vol = filtered_df[col].std() * np.sqrt(252)
        
        # Handle division by zero in Sharpe ratio
        if annualized_vol > 0:
            sharpe_ratio = (annualized_mean) / annualized_vol
        else:
            sharpe_ratio = np.nan


        wealth = (1 + filtered_df[col]).cumprod()
        running_max = (wealth.cummax())
        drawdown = (wealth - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics[col] = {
            'Annualized Mean': annualized_mean,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    return pd.DataFrame(metrics).T


def get_famafrench_daily(name, start=None, end=None):
    """
    Download and parse a Ken French daily factor dataset from Dartmouth.

    Parameters:
    -----------
    name (str): Dataset name without the _CSV suffix
        (e.g. 'F-F_Momentum_Factor_daily', 'F-F_Research_Data_Factors_daily')
    start (str, optional): Start date in 'YYYY-MM-DD' format for filtering
    end (str, optional): End date in 'YYYY-MM-DD' format for filtering

    Returns:
    --------
    pd.DataFrame: Daily factor returns indexed by date
    """
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    raw = requests.get(url, timeout=30).content
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    lines = [ln.rstrip(",") for ln in text.splitlines()]
    date_re = re.compile(r"^\s*\d{8}\s*,")
    data_rows = [i for i, ln in enumerate(lines) if date_re.match(ln)]
    start_i, end_i = data_rows[0], data_rows[-1] + 1
    header = [c.strip() or "Val" for c in lines[start_i - 1].split(",")]

    csv_text = ",".join(["Date"] + header[1:]) + "\n" + "\n".join(lines[start_i:end_i])
    df = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=[0], date_format="%Y%m%d")
    if start is not None or end is not None:
        df = df.loc[start:end]
    return df


def split_data(data, train_start_date, train_end_date, test_start_date, test_end_date):
    """
    Splits data into train and test sets

    Parameters:
    -----------
    data (pd.DataFrame): DataFrame containing data
    train_start_date (str): Start date in 'YYYY-MM-DD' format for training set
    train_end_date (str): End date in 'YYYY-MM-DD' format for training set
    test_start_date (str): Start date in 'YYYY-MM-DD' format for test set
    test_end_date (str): End date in 'YYYY-MM-DD' format for test set

    Returns:
    --------
    train_data (pd.DataFrame): DataFrame containing train data
    test_data (pd.DataFrame): DataFrame containing test data
    """

    # Splitting data into train and test sets
    train_data = data.loc[train_start_date:train_end_date]
    test_data = data.loc[test_start_date:test_end_date]
    return train_data, test_data


def regression_analysis(Y_data, X_data):
    """
    Generic regression analysis function, useful for both asset and factor regression

    Parameters:
    -----------
    Y_data (pd.DataFrame): Dependent variable
    X_data (pd.DataFrame): Independent variables

    Returns:
    --------
    pd.DataFrame: DataFrame containing regression results

    """
    # Initializing results dictionary

    results = {}

    # Converting Series to DataFrame if necessary
    if isinstance(X_data, pd.Series):
        x_df = X_data.to_frame()
    else:
        x_df = X_data.copy()

    if isinstance(Y_data, pd.Series):
        Y_df = Y_data.to_frame()
    else:
        Y_df = Y_data.copy()

    # Adding a constant for intercept
    X_const = sm.add_constant(x_df)

    # Common indices

    common_idx = Y_data.index.intersection(X_const.index)

    # Iterating through each column in Y_data
    for col in Y_df.columns:
        Y = Y_df[col].loc[common_idx].dropna()
        X = X_const.loc[Y.index]
        model = sm.OLS(Y, X).fit()
        results[col] = {
            'Alpha': model.params["const"],
            "Alpha-t": model.tvalues["const"],
            'Betas': model.params.drop("const").round(6).to_dict(),
            "Betas-t": model.tvalues.drop("const").round(6).to_dict(),
            'R-squared': model.rsquared,
            'R-squared Adj': model.rsquared_adj,
            'F-stat': model.fvalue,
        }
    return pd.DataFrame(results).T
    
def univariate_rolling_regression(Y_data, X_data, window=252):
    """
    Rolling OLS regression for Univariate Regression

    Parameters:
    -----------
    Y_data (pd.DataFrame): Dependent variable (Factor Returns/Portfolio Returns)
    X_data (pd.DataFrame or pd.Series): Independent variable (Market Returns/Factor Returns)
    window (int): Rolling window length (e.g. 252)

    Returns:
    --------
    Returns:
    pd.DataFrame: Time series of rolling alpha, betas, and R-squared
    """

     # Converting Series to DataFrame if necessary

    if isinstance(X_data, pd.Series):
        X_df = X_data.to_frame()
    else:
        X_df = X_data.copy()

    if isinstance(Y_data, pd.Series):
        Y_df = Y_data.to_frame()
    else:
        Y_df = Y_data.copy()

    # Common indices
    common_idx = Y_df.index.intersection(X_df.index)

    # Align data to common indices
    Y_aligned = Y_df.loc[common_idx]
    X_aligned = X_df.loc[common_idx]

    results = []

    for col in Y_aligned.columns:
        for end in range(window, len(common_idx)):
            y_window = Y_aligned[col].iloc[end - window:end]
            # Handle both Series and DataFrame for X_aligned
            if isinstance(X_aligned, pd.DataFrame):
                x_window_data = X_aligned.iloc[end - window:end].values
                if x_window_data.ndim > 1 and x_window_data.shape[1] == 1:
                    x_window_data = x_window_data.flatten()
                x_window = sm.add_constant(x_window_data)
            else:
                x_window = sm.add_constant(X_aligned.iloc[end - window:end])

            model = sm.OLS(y_window, x_window).fit()

            row = {
                'Date': Y_aligned.index[end],
                'Factor': col,
                'Alpha': model.params['const'],
                'Beta': model.params.iloc[1],
                'R-squared': model.rsquared,
                
            }
            results.append(row)

    # Check if results is empty (window too large for data)
    if len(results) == 0:
        print(f"Warning: No results generated. Window size ({window}) is larger than available data ({len(common_idx)} days).")
        print(f"Consider reducing window size to <= {len(common_idx)}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['Factor', 'Alpha', 'Beta', 'R-squared']).set_index(pd.DatetimeIndex([]))
       
    return pd.DataFrame(results).set_index('Date')


def multivariate_rolling_regression(Y_data, X_data, window=252):
    """
    Rolling OLS regression for Multivariate Regression (multiple factors)
    
    Parameters:
    -----------
    Y_data (pd.DataFrame): Dependent variable (Portfolio Returns) - each column is a portfolio
    X_data (pd.DataFrame): Independent variables (Factors) - columns are factors (MKT_RF, SMB, HML, MOM)
    window (int): Rolling window length (e.g. 252)
    
    Returns:
    --------
    dict: Dictionary where keys are portfolio names, values are DataFrames with:
          - Date index
          - Columns: Alpha, Beta_MKT_RF, Beta_SMB, Beta_HML, Beta_MOM, R-squared
    """
    # Converting Series to DataFrame if necessary
    if isinstance(X_data, pd.Series):
        X_df = X_data.to_frame()
    else:
        X_df = X_data.copy()
    
    if isinstance(Y_data, pd.Series):
        Y_df = Y_data.to_frame()
    else:
        Y_df = Y_data.copy()
    
    # Common indices
    common_idx = Y_df.index.intersection(X_df.index)
    
    # Align data to common indices
    Y_aligned = Y_df.loc[common_idx]
    X_aligned = X_df.loc[common_idx]
    
    # Get factor names
    factor_names = list(X_aligned.columns)
    
    # Dictionary to store results for each portfolio
    portfolio_results = {}
    
    # Iterate over each portfolio (column in Y_data)
    for portfolio_name in Y_aligned.columns:
        results = []
        
        for end in range(window, len(common_idx)):
            # Get rolling window data
            y_window = Y_aligned[portfolio_name].iloc[end - window:end]
            x_window = X_aligned.iloc[end - window:end]
            
            # Add constant for intercept
            x_window_const = sm.add_constant(x_window)
            
            # Fit OLS model
            model = sm.OLS(y_window, x_window_const).fit()
            
            # Extract coefficients
            row = {
                'Date': Y_aligned.index[end],
                'Portfolio': portfolio_name,
                'Alpha': model.params['const'],
            }
            
            # Add beta for each factor
            for factor in factor_names:
                if factor in model.params.index:
                    row[f'Beta_{factor}'] = model.params[factor]
                else:
                    row[f'Beta_{factor}'] = np.nan
            
            row['R-squared'] = model.rsquared
            
            results.append(row)
        
        # Check if results is empty
        if len(results) == 0:
            print(f"Warning: No results generated for {portfolio_name}. Window size ({window}) is larger than available data ({len(common_idx)} days).")
            portfolio_results[portfolio_name] = pd.DataFrame(columns=['Date', 'Portfolio', 'Alpha'] + [f'Beta_{f}' for f in factor_names] + ['R-squared']).set_index('Date')
        else:
            # Convert to DataFrame
            df_results = pd.DataFrame(results)
            df_results = df_results.set_index('Date')
            portfolio_results[portfolio_name] = df_results
    
    return portfolio_results

def process_macro_signals(macro_df, window = 252):
    """
    Transform raw macro data into tactical allocation signals

    Parameters:
    -----------
    macro_df (pd.DataFrame): DataFrame containing macro data
    window (int): Rolling window length (e.g. 252)

    Returns:
    --------
    pd.DataFrame: Processed signals with normalization
    """

    signals = pd.DataFrame(index = macro_df.index)

    # 1. Yield Curve Slope
    if 'T10Y2Y' in macro_df.columns:
        signals['Yield_Curve_Slope'] = macro_df['T10Y2Y']
    
    # 2. CPI Inflation YOY
    # CPI is monthly data - YoY requires 12 months
    # Resample to monthly, calculate 12-month change, then forward-fill to daily
    if 'CPIAUCSL' in macro_df.columns:
        cpi_monthly = macro_df['CPIAUCSL'].resample('ME').last() 
        cpi_yoy = cpi_monthly.pct_change(12) * 100 
        signals['CPI_YoY'] = cpi_yoy.reindex(macro_df.index, method = 'ffill')
    
    #3. VIX
    if 'VIXCLS' in macro_df.columns:
        signals['VIX'] = macro_df['VIXCLS']
    
    normalized = (signals - signals.rolling(window).mean())/signals.rolling(window).std()
    normalized.columns = [f'{col}_norm' for col in signals.columns]
   
    return pd.concat([signals, normalized], axis = 1)


def generate_tactical_views(macro_signals_df, tactical_results, factors_list, consensus_window = 63):
    """
    Translates current macro Z-scores into Annualized Expected Factor Returns.

    Parameters:
    -----------
    macro_signals_df (pd.DataFrame): DataFrame containing macro signals
    tactical_results (pd.DataFrame): DataFrame containing tactical results
    factors_list (list): List of factors
    consensus_window (int): Number of days to use for consensus (default is 63)

    Returns:
    --------
    pd.DataFrame: DataFrame containing tactical views
    """
    # Create empty dataframe for views
    views = pd.DataFrame(index=factors_list, columns=['Expected_Return_Ann', 'Alpha_Ann', 'R_squared', 'Signal_Strength'])
    
    # Grab the mean of the last 63 days of macro signals (the 'current' state)
    current_signals = macro_signals_df.tail(consensus_window).mean()
    
    for factor in factors_list:
        if factor in tactical_results.index:
            # Get results from regression function results
            alpha = tactical_results.loc[factor, 'Alpha']
            betas_dict = tactical_results.loc[factor, 'Betas']
            
            # Start with Alpha (Intercept)
            expected_daily_return = alpha
            
            # Add Beta * Current_Z_Score for each macro variable
            for macro_var, beta in betas_dict.items():
                if macro_var in current_signals.index:
                    expected_daily_return += beta * current_signals[macro_var]
            
            # 1. Annualize the expected return (Daily * 252)
            # This makes the Q vector in Black-Litterman readable (e.g., 0.05 for 5%)
            views.loc[factor, 'Expected_Return_Ann'] = expected_daily_return * 252
            views.loc[factor, 'Alpha_Ann'] = alpha * 252
            views.loc[factor, 'R_squared'] = tactical_results.loc[factor, 'R-squared']
            
            # 2. Measure Signal Strength (Sum of absolute Z-score impacts)
            # Useful for explaining why a view is large or small
            views.loc[factor, 'Signal_Strength'] = abs(expected_daily_return - alpha) / abs(alpha) if alpha != 0 else 0
            
    return views


def weight_calc(stock_list, factor_list, start_date, end_date, test_start_date, test_end_date):
    """
    Calculate the weights of the stocks in the portfolio. Function falls back to info if historical shares fail

    Parameters:
    -----------
    stock_list (list): List of stock tickers
    factor_list (list): List of factor names (Excluding the Market Factor)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    test_start_date (str): Start date in 'YYYY-MM-DD' format for test set
    test_end_date (str): End date in 'YYYY-MM-DD' format for test set
    
    Returns:
    --------
    pd.DataFrame: DataFrame containing the weights of the stocks in the portfolio
    """
    market_cap = []

    for stock in stock_list:
        ticker = yf.Ticker(stock)
        
        # 1. Get Actual (Unadjusted) Price - Critical for Mkt Cap
        # Use a small window ending at end_date to ensure we get a price
        price_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if price_data.empty:
            raise ValueError(f"No price data for {stock} at {end_date}")
        closing_price = price_data.iloc[-1]['Close']
        
        # --- SPLIT CHECK (TEST PERIOD) ---
        splits = ticker.splits
        split_in_test = False

        if not splits.empty:
            split_in_test = splits.loc[
                (splits.index >= test_start_date) &
                (splits.index <= test_end_date)
            ].any()

        # --- SHARES LOGIC ---
        if split_in_test:
            shares = ticker.info.get("sharesOutstanding")

        else:
            shares_series = ticker.get_shares_full(
                start=start_date,
                end=end_date
            )

            if shares_series is not None and not shares_series.empty:
                shares = shares_series.iloc[-1]
            else:
                shares = ticker.info.get("sharesOutstanding")

        if shares is None:
            raise ValueError(f"No shares available for {stock}")

        market_cap.append(shares * closing_price)

    # 3. Build the Stock DataFrame
    mkt_cap_df = pd.DataFrame(market_cap, index=stock_list, columns=['Market Cap'])
    mkt_cap_df['Total Market Cap'] = mkt_cap_df['Market Cap'].sum()
    mkt_cap_df['Weight'] = mkt_cap_df['Market Cap'] / mkt_cap_df['Total Market Cap']

    # 4. Append Factors (Ensures they are at the end of the vector)
    for factor in factor_list:
        mkt_cap_df.loc[factor, 'Market Cap'] = 0
        mkt_cap_df.loc[factor, 'Total Market Cap'] = 0
        mkt_cap_df.loc[factor, 'Weight'] = 0

    return mkt_cap_df

def calculate_covariance_matrices(returns_df, weights_df, annualize=True):
    """
    Calculate covariance matrix using both conventional and Ledoit-Wolf shrinkage methods.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame with returns (rows = dates, columns = assets/factors)
        Should already be excess returns
    annualize : bool, default=True
        If True, annualizes covariance by multiplying by 252 (trading days)
        If False, returns daily covariance
    weights_df : pd.DataFrame
        Contains output of weight_calc function, as it has desired order of assets/factors
    
    Returns:
    --------
    dict: Dictionary containing:
        - 'conventional': pd.DataFrame - Standard sample covariance matrix
        - 'ledoit_wolf': pd.DataFrame - Ledoit-Wolf shrinkage covariance matrix
        - 'shrinkage_factor': float - The shrinkage intensity (0-1)
    
    Notes:
    ------
    - Ledoit-Wolf shrinkage helps reduce estimation error and improve robustness
    - Shrinkage factor closer to 1 means more shrinkage toward identity matrix
    - Shrinkage factor closer to 0 means less shrinkage (closer to sample covariance)
    """
    # Drop any NaN rows to ensure clean calculation, while maintaining the order of assets/factors
    ordered_assets = weights_df.index.tolist()
    returns_clean = returns_df[ordered_assets].dropna()
    
    # 1. Conventional Covariance (Sample Covariance)
    cov_conventional = returns_clean.cov()
    
    # 2. Ledoit-Wolf Shrinkage Covariance
    # LedoitWolf expects numpy array (n_samples, n_features)
    returns_array = returns_clean.values
    
    # Fit Ledoit-Wolf estimator
    lw = LedoitWolf(store_precision = False)
    lw.fit(returns_array)
    
    # Get shrinkage covariance matrix
    cov_ledoit_wolf = pd.DataFrame(
        lw.covariance_,
        index=returns_clean.columns,
        columns=returns_clean.columns
    )
    
    # Get shrinkage factor (alpha in Ledoit-Wolf paper)
    shrinkage_factor = lw.shrinkage_
    
    # Annualize if requested
    if annualize:
        cov_conventional = cov_conventional * 252
        cov_ledoit_wolf = cov_ledoit_wolf * 252
    
    return {
        'conventional': cov_conventional,
        'ledoit_wolf': cov_ledoit_wolf,
        'shrinkage_factor': shrinkage_factor
    }

def lambda_calc(market_returns_df):
    """
    Calculate the lambda value for the market returns

    Parameters:
    -----------
    market_returns_df (pd.DataFrame): DataFrame containing the market returns

    Returns:
    --------
    dict : Dictionary containing:
        - 'lambda_market': float - The lambda value for the market investor (annualized mean / annualized variance)
        - 'lambda_trustee': float - The lambda value for the trustee investor (2x market)
        - 'lambda_kelly': float - The lambda value for the kelly investor (1)
    """
    # Calculate the lambda value for the market returns
    lambda_mean = market_returns_df.mean() * 252
    lambda_var = market_returns_df.var() * 252
    lambda_val_market = round(lambda_mean / lambda_var, 4)
    lambda_val_trustee = round(lambda_val_market * 2, 4)
    lambda_val_kelly = round(1, 4)

    return {
        'lambda_market': float(lambda_val_market),
        'lambda_trustee': float(lambda_val_trustee),
        'lambda_kelly': float(lambda_val_kelly)
    }

def generate_P_matrix(sigma_df, factor_list):
    """ 
    Generate P matrix from covariance matrix and factor list
    
    Parameters:
    -----------
    sigma_df (pd.DataFrame): Covariance matrix
    factor_list (list): List of factors
    
    Returns:
    --------
    P (pd.DataFrame): P matrix
    
    Notes:
    ------
    - P matrix is a matrix of factor returns
"""
    num_views = len(factor_list)
    num_assets = len(sigma_df.columns)

    P = np.zeros((num_views, num_assets))

    for i, factor in enumerate(factor_list):
        idx = sigma_df.columns.get_loc(factor)
        P[i, idx] = 1

    return P


def meucci_black_litterman(mu_prior, sigma_prior, P, Q, omega, tau):
    """
    Calculate posterior returns using Meucci's Black-Litterman model
    
    Parameters:
    -----------
    mu_prior : np.ndarray
        Prior mean returns
    sigma_prior : np.ndarray
        Prior covariance matrix
    P : np.ndarray 
        Matrix of factor exposures
    Q : np.ndarray
        Vector of expected returns
    omega : np.ndarray
        Vector of view confidence levels
    tau : float
        Risk aversion parameter
    
    Returns:
    --------
    np.ndarray: Posterior mean returns
    np.ndarray: Posterior covariance matrix
    """
    # Ensure inputs are numpy arrays
    mu_prior = np.array(mu_prior).reshape(-1, 1)
    sigma_prior = np.array(sigma_prior)
    P = np.array(P)                         
    Q = np.array(Q).reshape(-1, 1)     
    omega = np.array(omega)

    # 1. Calculate the Middle Term
    # This term captures the impact of views on the covariance matrix
    middle_term = np.linalg.inv(P.dot(tau * sigma_prior).dot(P.T) + omega)
    
    # 2. Update the Mean (mu_BL)
    # Adjustment = Prior_Cov * Pick_T * View_Precision * (View_Diff)
    mu_bl = mu_prior + (tau * sigma_prior).dot(P.T).dot(middle_term).dot(Q - P.dot(mu_prior))
    
    # 3. Update the Covariance (sigma_BL)
    # This reflects the reduction in uncertainty after incorporating views
    sigma_bl = (1 + tau) * sigma_prior - (tau**2 * sigma_prior).dot(P.T).dot(middle_term).dot(P).dot(sigma_prior)
    
    # Flatten the arrays to ensure they are 1D as expected by the optimizer
    return mu_bl.flatten(), sigma_bl
   
def optimize_mean_variance(mu, sigma, lam, asset_names, factor_names, factor_weight_lower = -1, factor_weight_upper = 1):
    """
    Optimize the mean-variance portfolio weights

    Parameters:
    -----------
    mu (np.array): Array containing returns
    sigma (np.array): Array containing covariance matrix
    lam (float): Lambda value for the risk-free rate
    asset_names (list): List of all asset names (Including factors)
    factor_names (list): List of factor names
    factor_weight_lower (float): Lower bound for factor weights (Must sum to 0 with upper bound)
    factor_weight_upper (float): Upper bound for factor weights (Must sum to 0 with lower bound)

    Returns:
    --------
    pd.Series: Series containing the optimized weights
    """

    n_total = len(mu)
    
    # Identify indices for stocks vs factors
    stock_indices = [i for i, name in enumerate(asset_names) if name not in factor_names]
    factor_indices = [i for i, name in enumerate(asset_names) if name in factor_names]
    
    # 1. Objective Function (Utility)
    def objective(w):
        # Negative utility for minimization
        return -(w @ mu - (lam / 2) * (w @ sigma @ w))

    # 2. Dynamic Constraints
    # Constraint 1: Sum of stock weights = 1
    # Constraint 2: Sum of factor weights = 0
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[stock_indices]) - 1},
        {'type': 'eq', 'fun': lambda w: np.sum(w[factor_indices]) - 0}
    ]
    
    # 3. Dynamic Bounds
    # Stocks: (0, 1) | Factors: as per the input
    bounds = []
    for i in range(n_total):
        if i in stock_indices:
            bounds.append((0, 1.0))
        else:
            bounds.append((factor_weight_lower, factor_weight_upper)) 
            
    # 4. Initial Guess
    init_guess = np.zeros(n_total)
    init_guess[stock_indices] = 1.0 / len(stock_indices) # Start with equal-weight stocks

    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    return res.x if res.success else None

def optimize_max_sharpe(mu, sigma, asset_names, factor_names, 
                        factor_weight_lower=-1, factor_weight_upper=1, rf=0):
    """
    Optimize the portfolio weights to maximize the Sharpe Ratio.

    Parameters:
    -----------
    mu (np.array): Array containing returns (usually BL posterior)
    sigma (np.array): Covariance matrix (Conventional or Ledoit-Wolf)
    asset_names (list): List of all asset names (including factors)
    factor_names (list): List of factor names
    factor_weight_lower (float): Lower bound for factor overlays
    factor_weight_upper (float): Upper bound for factor overlays
    rf (float): Risk-free rate (defaults to 0 for excess returns)

    Returns:
    --------
    np.array: Optimized weights (or None if optimization fails)
    """

    n_total = len(mu)
    
    # Identify indices for stocks vs factors
    stock_indices = [i for i, name in enumerate(asset_names) if name not in factor_names]
    factor_indices = [i for i, name in enumerate(asset_names) if name in factor_names]
    
    # 1. Objective Function (Negative Sharpe Ratio)
    def objective(w):
        portfolio_return = w @ mu
        # Variance calculation (w'Σw)
        portfolio_var = w @ sigma @ w
        # Volatility calculation (sqrt(Variance))
        portfolio_vol = np.sqrt(max(portfolio_var, 1e-10))
        
        sharpe_ratio = (portfolio_return - rf) / portfolio_vol
        # Minimize negative Sharpe to maximize actual Sharpe
        return -sharpe_ratio

    # 2. Dynamic Constraints
    # Constraint 1: Stock weights must sum to 1
    # Constraint 2: Factor overlay weights must sum to 0
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[stock_indices]) - 1},
        {'type': 'eq', 'fun': lambda w: np.sum(w[factor_indices]) - 0}
    ]
    
    # 3. Dynamic Bounds
    bounds = []
    for i in range(n_total):
        if i in stock_indices:
            bounds.append((0, 1.0))  # Stocks: No short selling
        else:
            bounds.append((factor_weight_lower, factor_weight_upper)) # Factors
            
    # 4. Initial Guess
    init_guess = np.zeros(n_total)
    init_guess[stock_indices] = 1.0 / len(stock_indices) # Equal-weight stocks as starting point

    # Numerical optimization using SLSQP solver
    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    return res.x if res.success else None


def optimize_erc(sigma, stock_names):
    """
    Optimize portfolio weights to achieve Equal Risk Contribution (ERC).
    Parameters:
    -----------
    sigma (np.array): Covariance matrix
    stock_names (list): List of all stock_names names (including factors)

    Returns:
    ----------
    pd.Series: Optimized weights

    Note: ERC is typically performed on the 'long-only' asset portion.
    """
    n = len(stock_names)
    
    # 1. Objective Function: Minimize the variance of Risk Contributions
    def objective(w):
        # Calculate Total Portfolio Volatility
        portfolio_vol = np.sqrt(w @ sigma @ w)
        
        # Calculate Marginal Contribution to Risk (MCR)
        # MCR = (Sigma * w) / Portfolio Volatility
        if portfolio_vol > 0:
            mcr = (sigma @ w) / portfolio_vol
        else:
            mcr = np.zeros(n)
        
        # Calculate Risk Contribution (RC)
        # RC = w * MCR
        rc = w * mcr
        
        # Calculate the 'Risk Imbalance'
        # We want RC[i] == RC[j] for all i, j. 
        # A fast way to write this is summing (RC[i] - RC[j])^2
        # np.tile helps create a matrix of differences
        diff = rc[:, np.newaxis] - rc
        return np.sum(np.square(diff))

    # 2. Constraints & Bounds
    # Constraint: Weights must sum to 1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: ERC requires long-only (0, 1) weights to stay stable
    bounds = [(0, 1.0) for _ in range(n)]
    
    # 3. Initial Guess: Equal weight (1/n)
    init_guess = np.repeat(1/n, n)

    # 4. Solve using Sequential Quadratic Programming (SLSQP)
    # This is recommended for non-linear risk budgeting
    res = minimize(objective, init_guess, method='SLSQP', 
                   bounds=bounds, constraints=cons, tol=1e-10)
    
    if not res.success:
        return None
        
    return pd.Series(res.x, index=stock_names)

def portfolio_return_metrics(portfolio_name,returns_df, weights, start_date, end_date, is_market_return = False):
    """
    Calculate financial metrics for selected period

    Parameters:
    -----------
    portfolio_name (str): Name of the portfolio
    returns_df (pd.DataFrame): DataFrame containing excess returns
    weights (pd.Series): Series containing portfolio weights
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    is_market_return (bool): Whether the returns are excess market returns
    Returns:
    --------
    dict: Dictionary containing calculated metrics

    """
    # Filter data for the specified date range
    filtered_df = returns_df.loc[start_date:end_date]

    # Check if filtered data is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for date range {start_date} to {end_date}")

    
    if is_market_return:
        portfolio_returns = filtered_df
    else:
        # Ensure weights and returns have same index
        common_index = filtered_df.columns.intersection(weights.index)
        # Calculate portfolio returns
        portfolio_returns = (filtered_df[common_index] * weights.loc[common_index]).sum(axis=1)

    # Calculate Metrics
    metrics = {}

    # Annualized Mean and Volatility
    annualized_mean = portfolio_returns.mean() *252
    annualized_vol = portfolio_returns.std() * np.sqrt(252)

    # Wealth / Cumulative Return
    n_days = len(portfolio_returns)
    wealth = (1+portfolio_returns).cumprod()
    total_return = float(wealth.iloc[-1]-1)
    ann_geometric_mean = ((1+total_return)**(252/n_days) - 1)

    # Diversification Ratio
    if not is_market_return:
        asset_vols = filtered_df[common_index].std() * np.sqrt(252)
        weighted_avg_vol = (asset_vols * weights.loc[common_index]).sum()
        diversification_ratio = weighted_avg_vol / annualized_vol if annualized_vol > 0 else np.nan
    else:
        diversification_ratio = np.nan
        
    # Handle division by zero in Sharpe ratio
    if annualized_vol > 0:
        sharpe_ratio = (annualized_mean) / annualized_vol
    else:
        sharpe_ratio = np.nan

    # Max Drawdown
    running_max = (wealth.cummax())
    drawdown = (wealth - running_max) / running_max
    max_drawdown = drawdown.min()


    metrics = {
        'Annualized Mean': annualized_mean,
        'Annualized Volatility': annualized_vol,
        'Annualized Geometric Mean': ann_geometric_mean,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Cumulative Return': total_return,
        'Diversification Ratio': diversification_ratio
        }

    df = pd.DataFrame.from_dict(metrics, orient='index', columns=[portfolio_name])
    cum_return_series = pd.Series(wealth-1 , index = wealth.index, name = portfolio_name)
    return df, cum_return_series


def calculate_portfolio_return_series(returns_df, weights, start_date, end_date, is_market_return=False):
    """
    Calculate portfolio return series (not metrics) for a given period.
    
    Parameters:
    -----------
    returns_df (pd.DataFrame): DataFrame containing excess returns
    weights (pd.Series or np.array): Series or array containing portfolio weights
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    is_market_return (bool): Whether the returns are excess market returns
    
    Returns:
    --------
    pd.Series: Series containing portfolio returns for the specified period
    """
    # Filter data for the specified date range
    filtered_df = returns_df.loc[start_date:end_date]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for date range {start_date} to {end_date}")
    
    if is_market_return:
        portfolio_returns = filtered_df
    else:
        # Convert weights to Series if it's an array
        if isinstance(weights, np.ndarray):
            weights = pd.Series(weights, index=filtered_df.columns)
        
        # Ensure weights and returns have same index
        common_index = filtered_df.columns.intersection(weights.index)
        # Calculate portfolio returns
        portfolio_returns = (filtered_df[common_index] * weights.loc[common_index]).sum(axis=1)
    
    return portfolio_returns


def generate_portfolio_returns_dataframe(portfolio_weights_dict, returns_df, start_date, end_date, 
                                         stock_list=None, use_stocks_only=True):
    """
    Generate a DataFrame of portfolio return series for multiple portfolios.
    Each column represents a portfolio return series, suitable for rolling regression.
    
    Parameters:
    -----------
    portfolio_weights_dict (dict): Dictionary where keys are portfolio names and values are weight Series/arrays
                                    OR a DataFrame where columns are portfolios and rows are assets
    returns_df (pd.DataFrame): DataFrame containing asset returns (stocks only or stocks + factors)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    stock_list (list): Optional list of stock names to filter weights (if use_stocks_only=True)
    use_stocks_only (bool): If True, only use stock weights (ignore factor weights)
    
    Returns:
    --------
    pd.DataFrame: DataFrame with dates as index and portfolio names as columns
                  Each column contains the portfolio return series
    """
    portfolio_returns_dict = {}
    
    # Handle both dict and DataFrame input formats
    if isinstance(portfolio_weights_dict, pd.DataFrame):
        # DataFrame format: columns are portfolios, rows are assets
        for portfolio_name in portfolio_weights_dict.columns:
            weights = portfolio_weights_dict[portfolio_name]
            
            # Filter to stocks only if requested
            if use_stocks_only and stock_list is not None:
                weights = weights.loc[stock_list]
            
            # Calculate portfolio returns
            portfolio_returns = calculate_portfolio_return_series(
                returns_df, weights, start_date, end_date, is_market_return=False
            )
            portfolio_returns_dict[portfolio_name] = portfolio_returns
    
    elif isinstance(portfolio_weights_dict, dict):
        # Dictionary format: keys are portfolio names, values are weight Series/arrays
        for portfolio_name, weights in portfolio_weights_dict.items():
            # Filter to stocks only if requested
            if use_stocks_only and stock_list is not None:
                if isinstance(weights, pd.Series):
                    weights = weights.loc[stock_list]
                elif isinstance(weights, np.ndarray):
                    # If array, need to map to stock_list indices
                    # This assumes weights array order matches returns_df columns
                    asset_names = returns_df.columns.tolist()
                    stock_indices = [asset_names.index(s) for s in stock_list if s in asset_names]
                    weights = pd.Series(weights[stock_indices], index=stock_list)
            
            # Calculate portfolio returns
            portfolio_returns = calculate_portfolio_return_series(
                returns_df, weights, start_date, end_date, is_market_return=False
            )
            portfolio_returns_dict[portfolio_name] = portfolio_returns
    
    else:
        raise ValueError("portfolio_weights_dict must be either a dict or pd.DataFrame")
    
    # Combine into DataFrame
    portfolio_returns_df = pd.DataFrame(portfolio_returns_dict)
    
    return portfolio_returns_df

def tracking_error_and_information_ratio(portfolios_dict, benchmark_weights, sigma_stocks, 
                                        portfolio_returns_annualized, benchmark_return_annualized):
    """
    Calculate tracking error and information ratio for multiple portfolios.
    
    Parameters:
    -----------
    portfolios_dict (dict): Dictionary where keys are portfolio names and values are portfolio weights (pd.Series)
    benchmark_weights (pd.Series): Benchmark weights (e.g., market cap weights)
    sigma_stocks (pd.DataFrame): Covariance matrix for stocks (annualized, Ledoit-Wolf)
    portfolio_returns_annualized (dict): Dictionary where keys are portfolio names and values are annualized returns (float)
    benchmark_return_annualized (float): Annualized benchmark return
    
    Returns:
    --------
    pd.DataFrame: DataFrame with columns:
        - 'Tracking Error (Annualized %)'
        - 'Portfolio Return (Annualized %)'
        - 'Benchmark Return (Annualized %)'
        - 'Excess Return (Annualized %)'
        - 'Information Ratio'
    """
    tracking_errors = {}
    information_ratios = {}
    
    # Calculate tracking error for each portfolio
    for portfolio_name, portfolio_weights in portfolios_dict.items():
        # Calculate active weights
        active_weights = portfolio_weights - benchmark_weights
        
        # Ensure alignment with covariance matrix
        active_weights_aligned = active_weights.loc[sigma_stocks.index]
        
        # Calculate tracking error: sqrt(w_active' @ Sigma @ w_active)
        tracking_error = np.sqrt(active_weights_aligned.values.T @ sigma_stocks.values @ active_weights_aligned.values)
        
        tracking_errors[portfolio_name] = tracking_error
        
        # Calculate information ratio
        if portfolio_name in portfolio_returns_annualized:
            excess_return = portfolio_returns_annualized[portfolio_name] - benchmark_return_annualized
            
            if tracking_error > 0:
                ir = excess_return / tracking_error
            else:
                ir = np.nan
            
            information_ratios[portfolio_name] = ir
    
    # Create comprehensive results table
    results_df = pd.DataFrame({
        'Tracking Error (Annualized %)': [te * 100 for te in tracking_errors.values()],
        'Portfolio Return (Annualized %)': [portfolio_returns_annualized[p] * 100 for p in portfolios_dict.keys()],
        'Benchmark Return (Annualized %)': benchmark_return_annualized * 100,
        'Excess Return (Annualized %)': [(portfolio_returns_annualized[p] - benchmark_return_annualized) * 100 
                                         for p in portfolios_dict.keys()],
        'Information Ratio': list(information_ratios.values())
    }, index=portfolios_dict.keys())
    
    return results_df