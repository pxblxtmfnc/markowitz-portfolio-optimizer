"""Portfolio metrics: returns, covariance, and performance calculations."""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily logarithmic returns from a price DataFrame."""
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")
    if prices.isnull().values.any():
        raise ValueError("Price DataFrame contains NaN values.")
    return np.log(prices / prices.shift(1)).dropna()


def annualize_returns(log_returns: pd.DataFrame) -> pd.Series:
    """Annualize mean daily log returns using 252 trading days."""
    if log_returns.empty:
        raise ValueError("Log returns DataFrame is empty.")
    return log_returns.mean() * TRADING_DAYS


def annualize_covariance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualize the covariance matrix of daily log returns."""
    if log_returns.empty:
        raise ValueError("Log returns DataFrame is empty.")
    return log_returns.cov() * TRADING_DAYS


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Calculate the expected annual return of a portfolio."""
    if len(weights) != len(expected_returns):
        raise ValueError("weights and expected_returns must have the same length.")
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculate the annualized portfolio volatility (standard deviation)."""
    if len(weights) != cov_matrix.shape[0]:
        raise ValueError("weights length must match covariance matrix dimensions.")
    variance = weights @ cov_matrix.values @ weights
    return float(np.sqrt(variance))


def sharpe_ratio(ret: float, vol: float, risk_free_rate: float = 0.0) -> float:
    """Compute the Sharpe Ratio given return, volatility and risk-free rate."""
    if vol <= 0:
        raise ValueError("Volatility must be positive to compute the Sharpe Ratio.")
    return (ret - risk_free_rate) / vol


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Return (annual_return, annual_volatility, sharpe_ratio) for a weight vector."""
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sr = sharpe_ratio(ret, vol, risk_free_rate)
    return ret, vol, sr
