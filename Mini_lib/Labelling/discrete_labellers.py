from ..Measures import Filters as fil
from ..Models import linearRegression as reg
from ..Labelling import common as com

import numpy as np
import pandas as pd

#! ==================================================================================== #
#! =============================== TRINARY LABELLERS ================================== #
class tripleBarrier_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "upper_barrier": [0.2, 0.5, 1, 2, 3, 5, 10],
                "lower_barrier": [0.2, 0.5, 1, 2, 3, 5, 10],
                "vertical_barrier": [5, 10, 15, 20, 25, 30],
                "window": [5, 10, 15, 20, 25, 30],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        upper_barrier: float,
        lower_barrier: float,
        vertical_barrier: int,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute volatility target =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().fillna(0)
        volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

        # ======= II. Initialize the labeled series and trade side =======
        labels_series = pd.Series(index=series.index, dtype=int)
        trade_side = 0

        # ======= III. Iterate through the price series =======
        for index in series.index:
            # III.1 Extract the future prices over the horizon
            start_idx = series.index.get_loc(index)
            end_idx = min(start_idx + vertical_barrier, len(series))
            future_prices = series.iloc[start_idx:end_idx]

            # III.2 Compute the range of future returns over the horizon
            max_price = future_prices.max()
            min_price = future_prices.min()

            max_price_index = future_prices.idxmax()
            min_price_index = future_prices.idxmin()

            max_return = (max_price - series.loc[index]) / series.loc[index]
            min_return = (min_price - series.loc[index]) / series.loc[index]

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
class lookForward_labeller(com.Labeller):
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window_lookForward": [5, 10, 15, 20, 25, 30],
                "min_trend_size": [5, 10, 15, 20, 25, 30],
                "volatility_threshold": [0.5, 1, 1.5, 2, 2.5, 3],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            params=params,
            n_jobs=n_jobs,
            )
    
    #?____________________________________________________________________________________ #
    def process_data(self, smoothing_method: str = None, window_smooth: int = None, lambda_smooth: float = None):
        if smoothing_method is None:
            processed_data = self.data
        elif smoothing_method == "ewma":
            processed_data = fil.ewma_smoothing(price_series=self.data, window=window_smooth, ind_lambda=lambda_smooth)
        elif smoothing_method == "average":
            processed_data = fil.average_smoothing(price_series=self.data, window=window_smooth)
        else:
            raise ValueError("Smoothing method not recognized")
        
        self.processed_data = processed_data
        
        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        window_lookForward: int,
        min_trend_size: int,
        volatility_threshold: float,
        smoothing_method: str,
        lambda_smooth: float,
        window_smooth: int,
    ):
        # ======= I. Prepare Series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()

        # ======= I. Significant look forward Label =======
        # ------- 1. Get the moving X days returns and the moving X days volatility -------
        Xdays_returns = (series.shift(-window_lookForward) - series) / series
        Xdays_vol = Xdays_returns.rolling(window=window_lookForward).std()

        # ------- 2. Compare the X days returns to the volatility  -------
        Xdays_score = Xdays_returns / Xdays_vol
        Xdays_label = Xdays_score.apply(lambda x: 1 if x > volatility_threshold else (-1 if x < -volatility_threshold else 0))

        # ------- 3. Eliminate the trends that are too small -------
        labels_series = com.trend_filter(label_series=Xdays_label, window=min_trend_size)

        return labels_series

