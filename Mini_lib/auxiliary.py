import numpy as np 
import pandas as pd

#! ==================================================================================== #
#! ================================== Series Filters ================================== #
def moving_average(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the moving average =======
    moving_avg = price_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    moving_avg = pd.Series(moving_avg, index=price_series.index)

    return moving_avg

#*____________________________________________________________________________________ #
def exponential_weightedMA(price_series: pd.Series, window: int, ind_lambda: float):
    # ======= I. Create the weights using a truncated exponential function =======
    weight_range = [(1 - ind_lambda) ** (i - 1) for i in range(1, window + 1)]
    weight_range.reverse()
    weight_range = np.array(weight_range)

    # ======= II. Perform the weighted moving average =======
    series = np.array(price_series)
    wma = get_weightedMA(series=series, weight_range=weight_range)

    # ======= III. Convert to pd.Series =======
    wma = pd.Series(wma, index=price_series.index)

    return wma


#! ==================================================================================== #
#! ================================ Helper Functions ================================== #
def get_weightedMA(series: pd.Series, weight_range: np.array):
        # ======= I. Check if the weights are valid =======
        values = np.array(series)
        values = values.astype("float64")
        wma = values.copy()

        if isinstance(weight_range, int):
            weights = np.array(range(1, weight_range + 1))
            rolling_window = weight_range
        else:
            weights = weight_range
            rolling_window = len(weight_range)

        # ======= II. Calculate the weighted moving average over a rolling window =======
        for i in range(0, len(values)):
            try:
                wma[i] = values[i - rolling_window + 1 : i + 1].dot(weights) / np.sum(weights)
            except:
                wma[i] = np.nan

        return wma

#*____________________________________________________________________________________ #
def get_volatility(price_series: pd.Series, window: int):

    returns_series = price_series.pct_change().fillna(0)
    volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

    return volatility_series

#! ==================================================================================== #
#! ================================ Labelling Process ================================= #
def trend_filter(label_series: pd.Series, window: int):
    # ======= I. Create an auxiliary DataFrame =======
    auxiliary_df = pd.DataFrame()
    auxiliary_df["label"] = label_series
    
    # ======= II. Create a group for each label and extract size =======
    auxiliary_df["group"] = (auxiliary_df["label"] != auxiliary_df["label"].shift()).cumsum()
    group_sizes = auxiliary_df.groupby("group")["label"].transform("size")

    # ======= III. Filter the labels based on the group size =======
    auxiliary_df["label"] = auxiliary_df.apply(lambda row: row["label"] if group_sizes[row.name] >= window else 0, axis=1)
    labels_series = auxiliary_df["label"]
    
    return labels_series


#! ==================================================================================== #
#! ============================= Gradient Descent Helper ============================== #
def adapt_learning_rate(learning_rate: float, loss: float, last_loss: float):
    new_rate = learning_rate
    if loss > last_loss:
        new_rate /= 2
    else:
        new_rate *= 1.05
    
    return new_rate

#*____________________________________________________________________________________ #
def early_stopping(loss: float, last_loss: float):
    # ======= I. Check the loss diference =======
    if last_loss == np.inf:
        return False
    
    loss_diff = np.abs(loss - last_loss)
    early_stop = False
    
    # ======= II. Check if the loss difference is small enough =======
    if loss_diff < 1e-5:
        early_stop = True
    
    return early_stop


#! ==================================================================================== #
#! ========================== Financial Performance Metrics =========================== #
def get_distribution(returns_series: pd.Series, frequence: str = "daily"):
    """
    - Expected Return: The annualized mean return, indicating average performance.
    - Volatility: Standard deviation of returns, representing total risk.
    - Downside Deviation: Standard deviation of negative returns, used in risk-adjusted metrics like Sortino Ratio.
    - Median Return: The median of returns, a measure of central tendency.
    - Skew & Kurtosis: Describe the distribution shape, with skew indicating asymmetry and kurtosis indicating tail heaviness.
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Compute the statistics =======
    expected_return = returns_series.mean() * adjusted_frequence
    volatility = returns_series.std() * np.sqrt(adjusted_frequence)
    downside_deviation = returns_series[returns_series < 0].std() * np.sqrt(adjusted_frequence) if returns_series[returns_series < 0].sum() != 0 else 0
    median_return = returns_series.median() * adjusted_frequence
    skew = returns_series.skew()
    kurtosis = returns_series.kurtosis()
    
    # ======= III. Store the statistics =======
    distribution_stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
    }
    
    return distribution_stats

#*____________________________________________________________________________________ #
def get_risk_measures(returns_series: pd.Series):
    """
    - Maximum Drawdown: Largest observed loss from peak to trough, a measure of downside risk.
    - Max Drawdown Duration: Longest period to recover from drawdown, indicating risk recovery time.
    - VaR 95 and CVaR 95: Value at Risk and Conditional Value at Risk at 95%, giving the maximum and average expected losses in worst-case scenarios.
    """
    # ======= I. Compute the Cumulative returns =======    
    cumulative_returns = (1 + returns_series).cumprod()
    
    # ======= II. Compute the statistics =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns_series.quantile(0.05)
    cvar_95 = returns_series[returns_series <= var_95].mean()
    
    # ======= III. Store the statistics =======
    risk_stats = {
        "drawdown": drawdown,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }
    
    return risk_stats

#*____________________________________________________________________________________ #
def get_market_sensitivity(returns_series: pd.Series, market_returns: pd.Series, frequence: str = "daily"):
    """
    - Beta: Sensitivity to market movements.
    - Alpha: Risk-adjusted return above the market return.
    - Upside/Downside Capture Ratios: Percent of market gains or losses captured by the investment.
    - Tracking Error: Volatility of return differences from the market.
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Compute the statistics =======
    # ------ Beta and Alpha (Jensens's)
    beta = returns_series.cov(market_returns) / market_returns.var()
    alpha = returns_series.mean() * adjusted_frequence - beta * (market_returns.mean() * adjusted_frequence)
    
    # ------ Capture Ratios
    upside_capture = returns_series[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
    downside_capture = returns_series[market_returns < 0].mean() / market_returns[market_returns < 0].mean()

    # ------ Tracking Error
    tracking_error = returns_series.sub(market_returns).std() * np.sqrt(adjusted_frequence)
    
    # ======= III. Store the statistics =======
    market_sensitivity_stats = {
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
    }
    
    return market_sensitivity_stats

#*____________________________________________________________________________________ #
def get_performance_measures(returns_series: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0, frequence: str = "daily"):
    """
    - Sharpe Ratio: Risk-adjusted returns per unit of volatility.
    - Sortino Ratio: Risk-adjusted return accounting only for downside volatility.
    - Treynor Ratio: Return per unit of systematic (market) risk.
    - Information Ratio: Excess return per unit of tracking error.
    - Sterling Ratio: Return per unit of average drawdown.
    - Calmar Ratio: Return per unit of maximum drawdown.
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Extract Statistics =======
    distribution_stats = get_distribution(returns_series, frequence)
    expected_return = distribution_stats["expected_return"]
    volatility = distribution_stats["volatility"]
    downside_deviation = distribution_stats["downside_deviation"]
    
    risk_stats = get_risk_measures(returns_series)
    drawdown = risk_stats["drawdown"]
    maximum_drawdown = risk_stats["maximum_drawdown"]
    
    market_sensitivity_stats = get_market_sensitivity(returns_series, market_returns, frequence)
    beta = market_sensitivity_stats["beta"]
    tracking_error = market_sensitivity_stats["tracking_error"]
    
    # ======= III. Compute the ratios =======
    # ------ Sharpe, Sortino, Treynor, and Information Ratios
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = expected_return / downside_deviation if downside_deviation != 0 else 0
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (expected_return - market_returns.mean() * adjusted_frequence) / tracking_error if tracking_error != 0 else 0

    # ------ Sterling, and Calmar Ratios
    average_drawdown = abs(drawdown[drawdown < 0].mean()) if drawdown[drawdown < 0].sum() != 0 else 0
    sterling_ratio = (expected_return - risk_free_rate) / average_drawdown if average_drawdown != 0 else 0
    calmar_ratio = expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
    
    # ======= IV. Store the statistics =======
    performance_stats = {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "sterling_ratio": sterling_ratio,
        "calmar_ratio": calmar_ratio,
    }
    
    return performance_stats, (distribution_stats, risk_stats, market_sensitivity_stats)