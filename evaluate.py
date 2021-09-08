from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def plot_residuals(y, yhat):
    """creates a residual plot"""
    res = yhat-y
    plt.scatter(y=res, x=y)
    plt.xlabel('y')
    plt.ylabel('residual')
    plt.show()

def regression_errors(y, yhat):
    '''returns the following values: sum of squared errors (SSE)
        explained sum of squares (ESS), total sum of squares (TSS)
        mean squared error (MSE), root mean squared error (RMSE), R-squared'''
    SSE = round(sum((yhat-y)**2),2)
    MSE = round(SSE/len(y),2)
    RMSE = round(sqrt(MSE),2)
    ESS = round(sum((yhat-y.mean())**2),2)
    TSS = round(ESS + SSE,2)
    R2 = round(ESS/TSS,2)
    
    print("SSE, ESS, TSS, MSE, RMSE, R2")
    return SSE, ESS, TSS, MSE, RMSE, R2
    
def baseline_mean_errors(y):
    """computes the SSE, MSE, and RMSE for the baseline model"""
    return regression_errors(y, np.zeros(len(y),) + y.mean())
    
def better_than_baseline(y, yhat):
    '''returns true if RMSE of model is less than baseline, otherwise false'''
    return True if baseline_mean_errors(y)[4] > regression_errors(y, yhat)[4] else False
    
