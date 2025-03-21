import sys
sys.path.append("../")
from Mini_lib import auxiliary as aux

import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ================================== Labellers ======================================= #
def tripleBarrier_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction =======
    upper_barrier = params["upper_barrier"]
    lower_barrier = params["lower_barrier"]
    vertical_barrier = params["vertical_barrier"]

    # ======= I. Compute volatility target =======
    p_series = price_series.dropna().copy()
    volatility_series = aux.get_volatility(price_series=p_series, window=vertical_barrier)

    # ======= II. Initialize the labeled series and trade side =======
    labels_series = pd.Series(index=p_series.index, dtype=int)
    trade_side = 0

    # ======= III. Iterate through the price series =======
    for index in p_series.index:
        # III.1 Extract the future prices over the horizon
        start_idx = p_series.index.get_loc(index)
        end_idx = min(start_idx + vertical_barrier, len(p_series))
        future_prices = p_series.iloc[start_idx:end_idx]

        # III.2 Compute the range of future returns over the horizon
        max_price = future_prices.max()
        min_price = future_prices.min()

        max_price_index = future_prices.idxmax()
        min_price_index = future_prices.idxmin()

        max_return = (max_price - p_series.loc[index]) / p_series.loc[index]
        min_return = (min_price - p_series.loc[index]) / p_series.loc[index]

        # III.3 Adjust the barrier thresholds with the volatility
        upper_threshold = upper_barrier * volatility_series.loc[index]
        lower_threshold = lower_barrier * volatility_series.loc[index]

        # III.4 Check if the horizontal barriers have been hit
        long_event = False
        short_event = False

        if trade_side == 1:  # Long trade
            if max_return > upper_threshold:
                long_event = True
            elif min_return < -lower_threshold:
                short_event = True

        elif trade_side == -1:  # Short trade
            if min_return < -upper_threshold:
                short_event = True
            elif max_return > lower_threshold:
                long_event = True

        else:  # No position held
            if max_return > upper_threshold:
                long_event = True
            elif min_return < -upper_threshold:
                short_event = True

        # III.5 Label based on the first event that occurs
        if long_event and short_event:  # If both events occur, choose the first one
            if max_price_index < min_price_index:
                labels_series.loc[index] = 1
            else:
                labels_series.loc[index] = -1

        elif long_event and not short_event:  # If only long event occurs
            labels_series.loc[index] = 1

        elif short_event and not long_event:  # If only short event occurs
            labels_series.loc[index] = -1

        else:  # If no event occurs (vertical hit)
            labels_series.loc[index] = 0

        # III.6 Update the trade side
        trade_side = labels_series.loc[index]

    return labels_series

#*____________________________________________________________________________________ #
def lookForward_labeller(price_series: pd.Series, params: dict):
    # ======= 0. Params extraction =======
    size_window_smooth = params["size_window_smooth"]
    lambda_smooth = params["lambda_smooth"]
    trend_size = params["trend_size"]
    volatility_threshold = params["volatility_threshold"]

    # ======= I. Prepare Series =======
    p_series = price_series.dropna().copy()
    ewma_series = aux.exponential_weightedMA(price_series=p_series, window=size_window_smooth, ind_lambda=lambda_smooth)

    # ======= I. Significant look forward Label =======
    # ------- 1. Get the moving X days returns and the moving X days volatility -------
    Xdays_returns = (ewma_series.shift(-size_window_smooth) - ewma_series) / ewma_series
    Xdays_vol = Xdays_returns.rolling(window=size_window_smooth).std()

    # ------- 2. Compare the X days returns to the volatility  -------
    Xdays_score = Xdays_returns / Xdays_vol
    Xdays_label = Xdays_score.apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

    # ------- 3. Eliminate the trends that are too small -------
    labels_series = aux.trend_filter(label_series=Xdays_label, window=trend_size)

    return labels_series