import sys
sys.path.append("../")
from Mini_lib import auxiliary as aux
from Mini_lib import Models as models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

#! ==================================================================================== #
#! ================================= Features ========================================= #
def average_features(
    price_series: pd.Series,
    window: int,
):
    # ======= I. Compute the different smoothed series =======
    rolling_average = price_series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))

    # ======= II. Convert to pd.Series and Center =======
    rolling_average = (pd.Series(rolling_average, index=price_series.index) / (price_series + 1e-8)) - 1
    
    # ======= III. Change Name =======
    rolling_average.name = f"average_{window}"

    return rolling_average

#*____________________________________________________________________________________ #
def quantile_features(
    price_series: pd.Series,
    window: int,
    quantile: float,
):
    # ======= I. Compute the rolling quantile =======
    returns_series = price_series.pct_change().dropna()
    rolling_quantile = returns_series.rolling(window=window + 1).apply(lambda x: np.quantile(x[:window], quantile))

    # ======= II. Convert to pd.Series and Center =======
    rolling_quantile = pd.Series(rolling_quantile, index=price_series.index)
    
    # ======= III. Change Name =======
    rolling_quantile.name = f"quantile_{quantile}_{window}"

    return rolling_quantile

#*____________________________________________________________________________________ #
def get_simple_TempReg(series: pd.Series):
    # ======= I. Fit the temporal regression =======
    X = np.arange(len(series))
    model = models.OLSRegression()
    model.fit(X, series)
    
    # ======= II. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    
    statistics, residuals = model.get_statistics()

    return intercept, coefficients, statistics,residuals

#*____________________________________________________________________________________ #
def linear_tempReg_features(
    price_series: pd.Series, 
    regression_window: int
):
    # ======= 0. Intermediate functions =======
    def compute_slope(series):
        _, coefficients, _, _ = get_simple_TempReg(series)
        slope = coefficients[0]
        
        return slope

    def compute_T_stats(series):
        _, _, statistics, _ = get_simple_TempReg(series)
        T_stats = statistics['T_stats'][0]
        
        return T_stats
    
    def compute_Pvalue(series):
        _, _, statistics, _ = get_simple_TempReg(series)
        P_value = statistics['P_values'][0]
        
        return P_value
    
    def compute_R_squared(series):
        _, _, statistics, _ = get_simple_TempReg(series)
        R_squared = statistics['R_squared']
        
        return R_squared

    # ======= I. Verify the price series is large enough =======
    if len(price_series) < regression_window:
        raise ValueError("Price series length must be greater than or equal to the regression window.")

    # ======= II. Compute the rolling regression statistics =======
    rolling_slope = price_series.rolling(window=regression_window + 1).apply(compute_slope, raw=False)
    rolling_tstat = price_series.rolling(window=regression_window + 1).apply(compute_T_stats, raw=False)
    rolling_pvalue = price_series.rolling(window=regression_window + 1).apply(compute_Pvalue, raw=False)
    rolling_r_squared = price_series.rolling(window=regression_window + 1).apply(compute_R_squared, raw=False)

    # ======= III. Convert to pd.Series and Unscale =======
    rolling_slope = pd.Series(rolling_slope, index=price_series.index) / (price_series + 1e-8)
    rolling_tstat = pd.Series(rolling_tstat, index=price_series.index)
    rolling_pvalue = pd.Series(rolling_pvalue, index=price_series.index)
    rolling_r_squared = pd.Series(rolling_r_squared, index=price_series.index)
    
    # ======= IV. Change Name =======
    rolling_slope.name = f"linear_slope_{regression_window}"
    rolling_tstat.name = f"linear_tstat_{regression_window}"
    rolling_pvalue.name = f"linear_pvalue_{regression_window}"
    rolling_r_squared.name = f"linear_r_squared_{regression_window}"

    return rolling_slope, rolling_tstat, rolling_pvalue, rolling_r_squared

#*____________________________________________________________________________________ #
def hurst_exponent_features(
    price_series: pd.Series, 
    power: int
):
    # ======= I. Initialize the variables =======
    prices_array = np.array(price_series)
    returns_array = prices_array[1:] / prices_array[:-1] - 1

    n = 2**power

    hursts = np.array([])
    tstats = np.array([])
    pvalues = np.array([])

    # ======= II. Compute the Hurst Exponent =======
    for t in np.arange(n, len(returns_array) + 1):
        data = returns_array[t - n : t]
        X = np.arange(2, power + 1)
        Y = np.array([])

        for p in X:
            m = 2**p
            s = 2 ** (power - p)
            rs_array = np.array([])

            for i in np.arange(0, s):
                subsample = data[i * m : (i + 1) * m]
                mean = np.average(subsample)
                deviate = np.cumsum(subsample - mean)
                difference = max(deviate) - min(deviate)
                stdev = np.std(subsample)
                rescaled_range = difference / stdev
                rs_array = np.append(rs_array, rescaled_range)

            Y = np.append(Y, np.log2(np.average(rs_array)))

        model = models.OLSRegression()
        model.fit(X, Y)
        
        hurst = model.coefficients[0]
        statistics, _ = model.get_statistics()
        tstat = statistics['T_stats'][0]
        pvalue = statistics['P_values'][0]
        
        hursts = np.append(hursts, hurst)
        tstats = np.append(tstats, tstat)
        pvalues = np.append(pvalues, pvalue)

    # ======= III. Convert to pd.Series and Center =======
    hursts = pd.Series([np.nan] * n + list(hursts), index=price_series.index) - 0.5
    tstats = pd.Series([np.nan] * n + list(tstats), index=price_series.index)
    pvalues = pd.Series([np.nan] * n + list(pvalues), index=price_series.index)

    tstats_mean = tstats.rolling(window=252).mean()
    tstats = tstats - tstats_mean

    pvalues_mean = pvalues.rolling(window=252).mean()
    pvalues = pvalues - pvalues_mean
    
    # ======= IV. Change Name =======
    hursts.name = f"hurst_exponent{power}"
    tstats.name = f"hurst_tstat_{power}"
    pvalues.name = f"hurst_pvalue_{power}"

    return hursts, tstats, pvalues


#! ==================================================================================== #
#! =============================== Features Analysis ================================== #
def feature_data(feature_series: pd.Series):
    # ======= I. Extract Basic Information =======
    data_type = feature_series.dtype
    
    missing_values = feature_series.isnull().sum()
    unique_values = feature_series.nunique()
    zero_values = (feature_series == 0).sum()
    negative_values = (feature_series < 0).sum()
    positive_values = (feature_series > 0).sum()
    
    # ======= II. Visualizing Basic Information =======
    print(f"Data Type: {data_type}")
    print(f"Missing Values: {missing_values}, Unique Values: {unique_values}")
    print(f"Zero Values: {zero_values}, Negative Values: {negative_values}, Positive Values: {positive_values}")
    
    # ======= III. Store Basic Information =======
    basic_info = {
        "Data Type": data_type,
        "Missing Values": missing_values,
        "Unique Values": unique_values,
        "Zero Values": zero_values,
        "Negative Values": negative_values,
        "Positive Values": positive_values
    }
    
    return basic_info

#*____________________________________________________________________________________ #
def feature_distribution(feature_series: pd.Series, feature_name: str = None):
    # ======= O. Feature name =======
    if feature_name is None:
        feature_name = "Feature"
    
    # ======= I. Extract Descriptive Statistics =======
    mean = feature_series.mean()
    median = feature_series.median()
    min_val = feature_series.min()
    max_val = feature_series.max()
    std_dev = feature_series.std()
    skewness = feature_series.skew()
    kurtosis = feature_series.kurtosis()
    
    # ======= II. Store Descriptive Statistics =======
    descriptive_df = pd.DataFrame({"Mean": [mean], "Median": [median], "Min": [min_val], "Max": [max_val], "Std. Dev": [std_dev], "Skewness": [skewness], "Kurtosis": [kurtosis]}, index=[feature_name])
    
    # ======= III. Visualizing Descriptive Statistics =======
    plt.figure(figsize=(17, 5))
    sns.histplot(feature_series, kde=True, bins=30, color="skyblue", stat="density", linewidth=0, label=f"{feature_name} Distribution")

    plt.axvline(mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')
    plt.axvspan(mean - std_dev, mean + std_dev, color='yellow', alpha=0.3, label='±1 Std Dev')

    plt.title(f'{feature_name} Distribution with Key Statistics')
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return descriptive_df

#*____________________________________________________________________________________ #
def feature_plot(feature_series: pd.Series, label_series: pd.Series, feature_name: str = None):
    # ======= I. Visualization of the feature against labels =======
    plt.figure(figsize=(17, 5))
    plt.plot(label_series.index, feature_series, label=feature_name, linewidth=2)

    for i, label in label_series.items():
        if label == 1:
            plt.scatter(i, 0, color='green', label='Reg: Upward Movement', s=10, zorder=5)
        elif label == 0:
            plt.scatter(i, 0, color='black', label='Reg: Neutral Movement', s=10, zorder=5)
        elif label == -1:
            plt.scatter(i, 0, color='red', label='Reg: Downward Movement', s=10, zorder=5)

    plt.title(f'Feature {feature_name} against Labels')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(False)
    plt.show()
    
    label_feature_df = pd.DataFrame({'label': label_series, feature_name: feature_series})
    plt.figure(figsize=(17, 5))
    sns.boxplot(x='label', y=feature_name, data=label_feature_df)
    plt.title(f'Boxplot of {feature_name} by Label')
    plt.xlabel('Labels')
    plt.ylabel(f'{feature_name} Values')
    plt.show()

#*____________________________________________________________________________________ #
def features_correlation(label_feature_df: pd.DataFrame):
    # ======= I. Correlation between the feature and the labels =======
    light_green = (0, 0.7, 0.3, 0.2)
    colors = [(0, 'green'), (0.5, light_green), (0.99, 'green'), (1, 'grey')]
    n_bins = 1000
    cmap_name = 'green_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    corr_matrix = label_feature_df.corr()
    plt.figure(figsize=(17, 5))
    sns.heatmap(corr_matrix, annot=True, cmap=cm, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

