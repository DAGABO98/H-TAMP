from typing import Tuple

import numpy as np
import numpy.typing as npt


NDArrayFloat = npt.NDArray[np.floating]


def MAE(pred: NDArrayFloat, true: NDArrayFloat) -> np.floating:
    return np.mean(np.abs(true - pred))


def MSE(pred: NDArrayFloat, true: NDArrayFloat) -> np.floating:
    return np.mean((true - pred) ** 2)


def RMSE(pred: NDArrayFloat, true: NDArrayFloat) -> np.floating:
    return np.sqrt(MSE(pred, true))


def SMAPE(pred: NDArrayFloat, true: NDArrayFloat) -> np.floating:
    return np.mean(np.abs(true - pred) / ((np.abs(true) + np.abs(pred)) / 2))


def MSPE(pred: NDArrayFloat, true: NDArrayFloat) -> np.floating:
    return np.mean(np.square((true - pred) / true))


def metric(pred: NDArrayFloat, true: NDArrayFloat) -> Tuple[np.floating, np.floating, np.floating]:
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)

    return mae, mse, rmse