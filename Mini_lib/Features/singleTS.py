from ..Measures import Filters as fil
from ..Measures import Momentum as mom
from ..Measures import Entropy as ent

from ..Features import common as com
from ..Models import linearRegression as reg

import numpy as np
import pandas as pd

#! ==================================================================================== #
#! ======================= Unscaled Smoothed-like Series Features ===================== #
class average_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "average" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_average = series.rolling(window=window + 1).apply(lambda x: np.mean(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_average = (pd.Series(rolling_average, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_average.name = f"average_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_average


#*____________________________________________________________________________________ #
class minimum_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "minimum" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_min = series.rolling(window=window + 1).apply(lambda x: np.min(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_min = (pd.Series(rolling_min, index=series.index) / (series + 1e-8)) - 1
        
        # ======= III. Change Name =======
        rolling_min.name = f"min_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_min
    
    

#! ==================================================================================== #
#! ========================== Returns Distribution Features =========================== #
class volatility_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "volatility" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_vol = returns_series.rolling(window=window + 1).apply(lambda x: np.std(x[:window]))

        # ======= II. Convert to pd.Series and Center =======
        rolling_vol = pd.Series(rolling_vol, index=series.index)
        
        # ======= III. Change Name =======
        rolling_vol.name = f"vol_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_vol


#*____________________________________________________________________________________ #
class quantile_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "quantile" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "quantile": [0.01, 0.05, 0.25, 0.75, 0.95, 0.99],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        quantile: float,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        returns_series = series.pct_change().dropna()
        rolling_quantile = returns_series.rolling(window=window + 1).apply(lambda x: np.quantile(x[:window], quantile))

        # ======= II. Convert to pd.Series and Center =======
        rolling_quantile = pd.Series(rolling_quantile, index=series.index)
        
        # ======= III. Change Name =======
        rolling_quantile.name = f"quantile_{quantile}_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_quantile



#! ==================================================================================== #
#! ============================= Series Trending Features ============================= #
class Z_momentum_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "Z_momentum" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        rolling_Z_momentum = series.rolling(window=window + 1).apply(lambda x: mom.get_Z_momentum(x[:window]))
        
        # ======= II. Convert to pd.Series and Center =======
        rolling_Z_momentum = pd.Series(rolling_Z_momentum, index=series.index)
        
        # ======= III. Change Name =======
        rolling_Z_momentum.name = f"Z_momentum_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}"

        return rolling_Z_momentum


#*____________________________________________________________________________________ #
class nonlinear_tempReg_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "nonlinear_tempreg" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "window": [5, 10, 30, 60, 120, 240],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        window: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= 0. Intermediate functions =======
        def compute_slope(series):
            _, coefficients, _, _ = mom.get_quad_TempReg(series)
            slope = coefficients[0]
            
            return slope
        
        def compute_acceleration(series):
            _, coefficients, _, _ = mom.get_quad_TempReg(series)
            acceleration = coefficients[1]
            
            return acceleration

        def compute_T_stats(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            T_stats = statistics['T_stats'][0]
            
            return T_stats
        
        def compute_Pvalue(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            P_value = statistics['P_values'][0]
            
            return P_value
        
        def compute_R_squared(series):
            _, _, statistics, _ = mom.get_quad_TempReg(series)
            R_squared = statistics['R_squared']
            
            return R_squared

        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()

        # ======= II. Compute the rolling regression statistics =======
        rolling_slope = series.rolling(window=window + 1).apply(compute_slope, raw=False)
        rolling_acceleration = series.rolling(window=window + 1).apply(compute_acceleration, raw=False)
        rolling_tstat = series.rolling(window=window + 1).apply(compute_T_stats, raw=False)
        rolling_pvalue = series.rolling(window=window + 1).apply(compute_Pvalue, raw=False)
        rolling_r_squared = series.rolling(window=window + 1).apply(compute_R_squared, raw=False)

        # ======= III. Convert to pd.Series and Unscale =======
        rolling_slope = pd.Series(rolling_slope, index=series.index) / (series + 1e-8)
        rolling_acceleration = pd.Series(rolling_acceleration, index=series.index) / (series + 1e-8)
        rolling_tstat = pd.Series(rolling_tstat, index=series.index)
        rolling_pvalue = pd.Series(rolling_pvalue, index=series.index)
        rolling_r_squared = pd.Series(rolling_r_squared, index=series.index)
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"nonlinear_slope_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_slope,
            f"nonlinear_acceleration_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_acceleration,
            f"nonlinear_tstat_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_tstat,
            f"nonlinear_pvalue_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_pvalue,
            f"nonlinear_r_squared_{window}_{smoothing_method}_{window_smooth}_{lambda_smooth}": rolling_r_squared,
        })

        return features_df


#*____________________________________________________________________________________ #
class hurst_exponent_feature(com.Feature):
    def __init__(
        self, 
        data: pd.Series, 
        name: str = "hurst_exponent" , 
        params: list = None, 
        n_jobs: int = 1
    ):
        # ======= 0. Initialize params if necessary =========
        if params is None:
            params = {
                "power": [3, 4, 5, 6, 7, 8],
                "smoothing_method": [None, "ewma", "average"],
                "window_smooth": [5, 10, 15, 20, 25, 30],
                "lambda_smooth": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }

        # ======= I. Get Base Model init =========
        super().__init__(
            data=data, 
            name=name,
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
    def get_feature(
        self,
        power: int,
        smoothing_method: str,
        window_smooth: int,
        lambda_smooth: float,
    ):
        # ======= I. Compute the different smoothed series =======
        series = self.process_data(smoothing_method=smoothing_method, window_smooth=window_smooth, lambda_smooth=lambda_smooth).dropna().copy()
        prices_array = np.array(series)
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

            model = reg.OLSRegression()
            model.fit(X, Y)
            
            hurst = model.coefficients[0]
            tstat = model.statistics['T_stats'][0]
            pvalue = model.statistics['P_values'][0]
            
            hursts = np.append(hursts, hurst)
            tstats = np.append(tstats, tstat)
            pvalues = np.append(pvalues, pvalue)

        # ======= III. Convert to pd.Series and Center =======
        hursts = pd.Series([np.nan] * n + list(hursts), index=series.index) - 0.5
        tstats = pd.Series([np.nan] * n + list(tstats), index=series.index)
        pvalues = pd.Series([np.nan] * n + list(pvalues), index=series.index)

        tstats_mean = tstats.rolling(window=252).mean()
        tstats = tstats - tstats_mean

        pvalues_mean = pvalues.rolling(window=252).mean()
        pvalues = pvalues - pvalues_mean
        
        # ======= IV. Change Name =======
        features_df = pd.DataFrame({
            f"hurst_exponent{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": hursts,
            f"hurst_tstat_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": tstats,
            f"hurst_pvalue_{power}_{smoothing_method}_{window_smooth}_{lambda_smooth}": pvalues,
        })
        
        return features_df

