import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import math

from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# plot_residuals(y, yhat): creates a residual plot
def plot_residuals(actual, predictions):
    residuals = actual - predictions
    plt.hlines(0, actual.min(), actual.max(),color='r', ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()

# regression errors 
def residuals(actual, predictions):
    return actual - predictions
# SSE
def sse(actual, predictions):
    return (residuals(actual, predictions) **2).sum()

# ESS
def ess(actual, predictions):
    return ((predictions - actual.mean()) ** 2).sum()

# TSS
def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

# MSE
def mse(actual, predictions):
    n = actual.shape[0]
    return sse(actual, predictions) / n

# RMSE
def rmse(actual, predictions):
    return math.sqrt(mse(actual, predictions))

# Variance explained
def r2_score(actual, predictions):
    return ess(actual, predictions) / tss(actual)


def regression_errors(actual, predictions):
    return pd.Series({
        'SSE': sse(actual, predictions),
        'ESS': ess(actual, predictions),
        'TSS': tss(actual),
        'MSE': mse(actual, predictions),
        'RMSE': rmse(actual, predictions),
    })

# Baseline mean errors
def baseline_mean_errors(actual):
    predictions = actual.mean()
    return pd.Series({
        'SSE': sse(actual, predictions),
        'MSE': mse(actual, predictions),
        'RMSE': rmse(actual, predictions),
    })

# Better than baseline
def better_than_baseline(actual, predictions):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predictions)
    return rmse_model < rmse_baseline
