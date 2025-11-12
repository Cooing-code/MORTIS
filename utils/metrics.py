import numpy as np

def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true)) * 100
    mspe = np.mean(np.square((pred - true) / true)) * 100
    return (mae, mse, rmse, mape, mspe)
