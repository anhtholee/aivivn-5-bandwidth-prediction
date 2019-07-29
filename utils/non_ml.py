# Some function for estimating the predictions using non-ML approaches
import numpy as np
import pandas as pd

def geo_mean(iterable):
    """
        Calculate geometric mean of a series
    """
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def moving_average(series, n, gmean=False):
    """
        Calculate average of last n observations
    """
    if gmean:
        return geo_mean(series[-n:])
    else:
        return np.average(series[-n:])
    
def moving_median(series, n):
    """
        Calculate median of last n observations
    """
    return np.median(series[-n:])

def moving_min(series, n):
    return np.amin(series[-n:])

# Based on this kernel: https://www.kaggle.com/safavieh/median-estimation-by-fibonacci-et-al-lb-44-9
def median_estimation(series, windows):
    """
        Estimate the predictions using median of medians in different window sizes
    """
    M = []
    start = series[:].nonzero()[0]
    n = len(series)
    if n - start[0] < windows[0]:
        res = series.iloc[start[0]+1:].median()
        return res
    for w in windows:
        if w > n - start[0]:
            break
        M.append(series.iloc[-w:].median())
    res = np.median(M)
    return res