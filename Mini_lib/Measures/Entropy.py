import numpy as np 
import pandas as pd

#! ==================================================================================== #
#! =========================== Series Entropy Functions =============================== #
def get_shannon_entropy(signs_series: pd.Series):
    # ======= I. Count the frequency of each symbol in the data =======
    _, counts = np.unique(signs_series, return_counts=True)

    # ======= II. Compute frequentist probabilities =======
    probabilities = counts / len(signs_series)

    # ======= III. Compute Shannon entropy =======
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

#*____________________________________________________________________________________ #
def get_gini_impurity(signs_series: pd.Series):
    """
    Computes Gini Impurity
    """
    classes, counts = np.unique(signs_series, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)
