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
    # Data processing
    'factor_return_metrics',
    'get_famafrench_daily',
    'split_data',
    # Regression
    'regression_analysis',
    'univariate_rolling_regression',
    'multivariate_rolling_regression',
    # Legacy macro-signal view pipeline (superseded, retained as documentation)
    'process_macro_signals',
    'generate_tactical_views',
    # Regime classification and conditional statistics (current view pipeline)
    'label_regimes',
    'episode_id',
    'count_episodes',
    'regime_snapshot',
    'occupancy',
    'association',
    'era_split',
    'test_drift',
    'conditional_stats',
    # Weights, covariance, risk aversion
    'weight_calc',
    'calculate_covariance_matrices',
    'lambda_calc',
    # Black-Litterman
    'generate_P_matrix',
    'meucci_black_litterman',
    # Optimizers
    'optimize_mean_variance',
    'optimize_max_sharpe',
    'optimize_erc',
    # Performance
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

# ============================================================================
# Regime classification
# ============================================================================

def label_regimes(df):
    """
    Regime labels. Every threshold is real-time: fixed economic
    constants, or an expanding-window statistic. No future information.

    Parameters:
    -----------
    df (pd.DataFrame): Monthly macro panel, already lagged to publication
        schedule. Must contain: cfnai, cpi_yoy, cpi_med (expanding median of
        cpi_yoy), real_ff, curve, vix, cpi_dir

    Returns:
    --------
    pd.DataFrame: Same index as `df`, with conditioner columns
        (growth, infl_lvl, policy), grid columns (grid_A, grid_B, grid_M) and
        diagnostic-only columns (curve_d, vol_d, dir_d)
    """
    out = pd.DataFrame(index=df.index)

    # --- Conditioners -------------------------------------------------
    # CFNAI is constructed so 0 = trend growth
    out['growth']   = np.where(df['cfnai']   < 0.0,           'below_trend',   'above_trend')
    # Inflation relative to its own expanding median (see note below)
    out['infl_lvl'] = np.where(df['cpi_yoy'] > df['cpi_med'], 'high',          'low')
    # Real fed funds: positive = restrictive
    out['policy']   = np.where(df['real_ff'] < 0.0,           'accommodative', 'restrictive')

    # --- Grids, per the pre-committed assignment ----------------------
    out['grid_A'] = out['growth'] + ' / ' + out['infl_lvl']   # HML
    out['grid_B'] = out['growth'] + ' / ' + out['policy']     # SMB
    out['grid_M'] = out['growth']                             # MOM (marginal)

    # --- Diagnostic only ----------------------------------------------
    out['curve_d'] = np.where(df['curve']   < 0.0,  'inverted', 'normal')
    out['vol_d']   = np.where(df['vix']     > 20.0, 'high',     'low')
    out['dir_d']   = np.where(df['cpi_dir'] > 0.0,  'rising',   'falling')
    return out


# ============================================================================
# Episode helpers
# ============================================================================

def episode_id(mask):
    """Label each contiguous run of True with a distinct integer id.

    Returns a Series aligned to `mask`, holding the run id where mask is True
    and NaN elsewhere. Used to group returns within an episode so that
    uncertainty can be computed across episodes rather than across months.
    """
    m = mask.astype(int)
    return (m.diff() == 1).cumsum().where(mask)


def count_episodes(mask):
    """Number of contiguous runs where `mask` is True.

    `.diff() == 1` counts every transition False -> True but cannot see a run
    that is already in progress at the first observation, so that case is
    added explicitly.
    """
    m = mask.astype(int)
    n = int((m.diff() == 1).sum())
    if len(m) and m.iloc[0] == 1:
        n += 1
    return n


# ============================================================================
# Regime verification helpers
# ============================================================================
# Each returns a DataFrame rather than printing, so output renders as a table
# and can be reused - exported, asserted on, or referenced downstream.

def regime_snapshot(period, regimes, macro, spec, sample=None):
    """Classification at a single date, with distance to each threshold.

    `spec` maps regime column -> (macro column, threshold, label_if_above).
    Threshold may be a constant or the name of a column (e.g. an expanding median).

    Margin is reported in the variable's own units and, more usefully, scaled by
    that variable's standard deviation - CFNAI is an index, inflation a decimal
    rate, so raw margins are not comparable across rows.
    """
    sample = macro if sample is None else sample
    rows = []
    for reg_col, (macro_col, thresh, _) in spec.items():
        val = macro.loc[period, macro_col]
        thr = macro.loc[period, thresh] if isinstance(thresh, str) else thresh
        sd  = sample[macro_col].std()
        margin = val - thr
        rows.append({
            'variable':    reg_col,
            'input':       macro_col,
            'value':       val,
            'threshold':   thr,
            'margin':      margin,
            'margin_/_sd': margin / sd if sd else np.nan,
            'label':       regimes.loc[period, reg_col],
        })
    out = pd.DataFrame(rows).set_index('variable')
    # within 0.25 sd of the threshold -> could flip on a data revision
    out['fragile'] = out['margin_/_sd'].abs() < 0.25
    return out


def occupancy(regimes, cols, n_total=None):
    """Months, independent episodes, and average episode length per state.

    `se_inflation` = sqrt(avg episode length), approximately the factor by which
    a month-count standard error understates true uncertainty. This is the
    quantity that motivates building Omega from episode-clustered errors.
    """
    n_total = len(regimes) if n_total is None else n_total
    rows = []
    for col in cols:
        for state in sorted(regimes[col].unique()):
            m = regimes[col] == state
            n_ep = count_episodes(m)
            avg = m.sum() / n_ep if n_ep else np.nan
            rows.append({
                'variable':     col,
                'state':        state,
                'months':       int(m.sum()),
                'pct':          round(100 * m.sum() / n_total),
                'episodes':     n_ep,
                'avg_ep_len':   round(avg, 1),
                'se_inflation': round(np.sqrt(avg), 2) if avg == avg else np.nan,
            })
    return pd.DataFrame(rows).set_index(['variable', 'state'])


def association(regimes, col_a, col_b, concordant_pairs):
    """Cross-classification of two regime axes, plus a strength measure.

    `cramers_v` is 0 under independence, 1 when one axis determines the other.
    For a 2x2 table it equals |phi|.

    `concordance` is the share of months in the economically paired states,
    named explicitly via `concordant_pairs` rather than read off the matrix
    diagonal - the diagonal depends on alphabetical sort order and pairs
    'high' with 'accommodative', which is the opposite of what is intended.
    """
    ct  = pd.crosstab(regimes[col_a], regimes[col_b])
    n   = ct.values.sum()
    exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    v   = np.sqrt(((ct.values - exp) ** 2 / exp).sum() / (n * (min(ct.shape) - 1)))
    summary = pd.Series({
        'n_months':    int(n),
        'cramers_v':   round(v, 3),
        'concordance': round(sum(ct.loc[a, b] for a, b in concordant_pairs) / n, 3),
        'concordance_if_independent': round(
            sum((ct.sum(axis=1)[a] / n) * (ct.sum(axis=0)[b] / n)
                for a, b in concordant_pairs), 3),
    })
    return ct, summary


def era_split(factor, regime_col, split_year, labels, sample, regimes):
    """Conditional means by regime state, computed separately within each era.

    Does a regime spread survive WITHIN each era, or does it only appear when
    eras are pooled? The latter would mean an era effect is being mistaken for
    a regime effect.

    Run on the FULL history (`calib_full` / `reg_calib_full`), not the restricted
    calibration window - the whole point is to inspect the period that was
    excluded. `sample` and `regimes` are explicit arguments here; the notebook
    cell binds them to the full-history objects.

    Marginal axes are used rather than the 2x2 cells: splitting four cells across
    two eras leaves eight groups, and the pre-1983 groups are too thin to read.
    """
    cut = pd.Period(f'{split_year}-01', 'M')
    states = sorted(regimes[regime_col].unique())
    rows = []

    for name, sub in [(labels[0], sample[sample.index < cut]),
                      (labels[1], sample[sample.index >= cut])]:
        lbl = regimes.loc[sub.index, regime_col]
        r = {'era': name, 'n_months': len(sub)}
        for state in states:
            m = lbl == state
            r[f'{state}_n']    = int(m.sum())
            r[f'{state}_mean'] = sub.loc[m, factor].mean() * 12 if m.sum() else np.nan
        rows.append(r)

    df = pd.DataFrame(rows).set_index('era')
    df['spread'] = df[f'{states[0]}_mean'] - df[f'{states[1]}_mean']
    return df


def test_drift(regimes_calib, regimes_test, view_period, grids):
    """Did the view-date regime persist through the out-of-sample window?

    POST-HOC ONLY. Views are formed once at `view_period`; this measures how
    much of the test window the resulting static view actually applied to.
    Never feeds calibration.
    """
    rows = []
    for g in grids:
        held  = regimes_calib.loc[view_period, g]
        match = regimes_test[g] == held
        rows.append({
            'grid':            g,
            'view_date_cell':  held,
            'months_matching': int(match.sum()),
            'months_total':    len(regimes_test),
            'pct_matching':    round(100 * match.mean()),
            'realised_states': ' | '.join(
                f'{k}: {v}' for k, v in regimes_test[g].value_counts().items()),
        })
    return pd.DataFrame(rows).set_index('grid')


# ============================================================================
# Conditional statistics with episode clustering
# ============================================================================

def conditional_stats(factor_returns, regime_labels, ann=12):
    """Per-regime performance statistics with episode-clustered standard errors.

    Parameters
    ----------
    factor_returns : Series of monthly factor returns
    regime_labels  : Series of regime state labels, aligned to factor_returns
    ann            : periods per year (12 for monthly)

    Returns
    -------
    DataFrame indexed by regime state.

    Notes
    -----
    `se_naive_ann` treats months as independent and is reported only to show how
    much precision that assumption manufactures. `se_cluster_ann` is computed
    across contiguous-episode means and is the figure that should inform Omega.
    """
    rows = []
    for state in sorted(regime_labels.unique()):
        mask = regime_labels == state
        r = factor_returns[mask]
        if len(r) == 0:
            continue

        ep = episode_id(mask)
        grouped = r.groupby(ep[mask])
        ep_mean = grouped.mean()
        ep_cum = grouped.apply(lambda x: (1.0 + x).prod() - 1.0)
        n_ep = int(len(ep_mean))

        rows.append({
            'state':          state,
            'n_months':       int(mask.sum()),
            'n_episodes':     n_ep,
            'avg_ep_len':     mask.sum() / n_ep if n_ep else np.nan,
            'mean_ann':       r.mean() * ann,
            'std_ann':        r.std(ddof=1) * np.sqrt(ann),
            'hit_rate':       (r > 0).mean(),
            'ep_min':         ep_cum.min(),
            'ep_max':         ep_cum.max(),
            'se_naive_ann':   r.std(ddof=1) / np.sqrt(len(r)) * ann,
            'se_cluster_ann': (ep_mean.std(ddof=1) / np.sqrt(n_ep) * ann
                               if n_ep > 1 else np.nan),
        })

    out = pd.DataFrame(rows).set_index('state')
    out['se_ratio'] = out['se_cluster_ann'] / out['se_naive_ann']
    return out


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
        View uncertainty. Either a 1-D vector of per-view variances (placed on
        the diagonal) or a full k x k matrix. A 1-D vector is diagonalised here:
        adding it raw would broadcast across rows of P tau Sigma P' and produce
        a near-singular matrix whose inverse is numerical noise.
    tau : float
        Prior uncertainty scaling

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
    if omega.ndim == 1:
        omega = np.diag(omega)

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
                                         portfolio_returns_annualized, benchmark_return_annualized,
                                         returns_df=None, start_date=None, end_date=None,
                                         periods_per_year=252):
    """
    Tracking error and information ratio, reported ex-ante and ex-post.

    Two tracking errors are computed because they answer different questions and
    must not be mixed:

      TE ex-ante  = sqrt(w_active' @ Sigma @ w_active), Sigma estimated on the
                    TRAINING window. A forecast of active risk.
      TE ex-post  = std(active return series) * sqrt(periods_per_year), computed
                    on the REALISED test-window returns. What actually happened.

    The information ratio is reported only against the ex-post tracking error,
    because its numerator (realised excess return) is ex-post. Dividing a
    realised excess return by a forecast risk is neither an ex-ante nor an
    ex-post IR, and is not reported here.

    `TE Ratio (Post/Ante)` is the diagnostic: above 1 means realised active risk
    exceeded the forecast. Expect it above 1 for portfolios whose weights were
    optimised on `sigma_stocks` itself - the optimiser exploits estimation error
    in Sigma, so the in-sample forecast is biased low.

    Parameters:
    -----------
    portfolios_dict (dict): {portfolio name: portfolio weights (pd.Series)}
    benchmark_weights (pd.Series): Benchmark weights (e.g. market cap weights)
    sigma_stocks (pd.DataFrame): Annualized covariance matrix for stocks (training window)
    portfolio_returns_annualized (dict): {portfolio name: realised annualized return (float)}
    benchmark_return_annualized (float): Realised annualized benchmark return
    returns_df (pd.DataFrame, optional): Test-window asset returns, dates x tickers.
        Required for the ex-post columns; if omitted they come back NaN.
        Must be on the same basis (excess vs total) as the annualized returns above.
    start_date, end_date (str, optional): Slice applied to `returns_df`
    periods_per_year (int): Annualization factor for the realised series (252 daily)

    Returns:
    --------
    pd.DataFrame indexed by portfolio name, with columns:
        - 'TE Ex-Ante (Annualized %)'
        - 'TE Ex-Post (Annualized %)'
        - 'TE Ratio (Post/Ante)'
        - 'Portfolio Return (Annualized %)'
        - 'Benchmark Return (Annualized %)'
        - 'Excess Return (Annualized %)'
        - 'IR Ex-Post'
    """
    if returns_df is not None and (start_date is not None or end_date is not None):
        returns_df = returns_df.loc[start_date:end_date]

    rows = {}

    for portfolio_name, portfolio_weights in portfolios_dict.items():
        # Active weights. Pandas aligns on index, so a ticker present in one leg
        # but not the other yields NaN - caught here rather than propagating
        # silently into a NaN tracking error.
        active_weights = portfolio_weights - benchmark_weights
        active_weights_aligned = active_weights.loc[sigma_stocks.index]
        if active_weights_aligned.isna().any():
            missing = list(active_weights_aligned.index[active_weights_aligned.isna()])
            raise ValueError(
                f"'{portfolio_name}': active weights are NaN for {missing}. "
                "Portfolio and benchmark weights must cover sigma_stocks.index."
            )

        # --- Ex-ante: forecast from the training-window covariance -----------
        te_ante = np.sqrt(
            active_weights_aligned.values.T @ sigma_stocks.values @ active_weights_aligned.values
        )

        # --- Ex-post: realised dispersion of the active return series --------
        te_post = np.nan
        if returns_df is not None:
            cols = [c for c in active_weights_aligned.index if c in returns_df.columns]
            missing_cols = [c for c in active_weights_aligned.index if c not in returns_df.columns]
            if missing_cols:
                raise ValueError(
                    f"'{portfolio_name}': returns_df has no columns for {missing_cols}."
                )
            active_series = (returns_df[cols] * active_weights_aligned[cols]).sum(axis=1)
            if len(active_series) > 1:
                te_post = active_series.std(ddof=1) * np.sqrt(periods_per_year)

        excess_return = (portfolio_returns_annualized[portfolio_name]
                         - benchmark_return_annualized)

        rows[portfolio_name] = {
            'TE Ex-Ante (Annualized %)':       te_ante * 100,
            'TE Ex-Post (Annualized %)':       te_post * 100,
            'TE Ratio (Post/Ante)':            te_post / te_ante if te_ante > 0 else np.nan,
            'Portfolio Return (Annualized %)': portfolio_returns_annualized[portfolio_name] * 100,
            'Benchmark Return (Annualized %)': benchmark_return_annualized * 100,
            'Excess Return (Annualized %)':    excess_return * 100,
            # IR against the ex-post TE only: both terms realised, same window
            'IR Ex-Post':                      excess_return / te_post if te_post > 0 else np.nan,
        }

    return pd.DataFrame(rows).T