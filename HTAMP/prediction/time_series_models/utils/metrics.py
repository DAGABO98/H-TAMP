from typing import Tuple

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def _to_float_array(x: ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def MAE(pred: ArrayLike, true: ArrayLike) -> float:
    pred_arr = _to_float_array(pred)
    true_arr = _to_float_array(true)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def MSE(pred: ArrayLike, true: ArrayLike) -> float:
    pred_arr = _to_float_array(pred)
    true_arr = _to_float_array(true)
    return float(np.mean((true_arr - pred_arr) ** 2))


def RMSE(pred: ArrayLike, true: ArrayLike) -> float:
    return float(np.sqrt(MSE(pred, true)))


def SMAPE(pred: ArrayLike, true: ArrayLike, eps: float = 1e-8) -> float:
    pred_arr = _to_float_array(pred)
    true_arr = _to_float_array(true)

    denominator = (np.abs(true_arr) + np.abs(pred_arr)) / 2.0
    denominator = np.maximum(denominator, eps)

    return float(np.mean(np.abs(true_arr - pred_arr) / denominator))


def MSPE(pred: ArrayLike, true: ArrayLike, eps: float = 1e-8) -> float:
    pred_arr = _to_float_array(pred)
    true_arr = _to_float_array(true)

    denominator = np.maximum(np.abs(true_arr), eps)
    return float(np.mean(np.square((true_arr - pred_arr) / denominator)))


def metric(pred: ArrayLike, true: ArrayLike) -> Tuple[float, float, float]:
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    return mae, mse, rmse