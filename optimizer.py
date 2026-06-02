"""Portfolio optimization using scipy.optimize.minimize (SLSQP)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

from metrics import (
    portfolio_return,
    portfolio_volatility,
    portfolio_performance,
)


def _base_constraints(n: int) -> list[dict]:
    """Weights must sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _bounds(n: int) -> list[tuple[float, float]]:
    """Each weight in [0, 1] (long-only)."""
    return [(0.0, 1.0)] * n


def _equal_weights(n: int) -> np.ndarray:
    """Equal-weight starting point for the optimizer."""
    return np.full(n, 1.0 / n)


def _validate_inputs(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> None:
    """Raise ValueError for mismatched or degenerate inputs."""
    if expected_returns is None or len(expected_returns) == 0:
        raise ValueError("expected_returns must be a non-empty pd.Series.")
    if cov_matrix is None or cov_matrix.empty:
        raise ValueError("cov_matrix must be a non-empty pd.DataFrame.")
    if len(expected_returns) != cov_matrix.shape[0]:
        raise ValueError(
            f"expected_returns length ({len(expected_returns)}) must match "
            f"cov_matrix dimensions ({cov_matrix.shape[0]})."
        )
    if not np.allclose(cov_matrix.values, cov_matrix.values.T):
        raise ValueError("cov_matrix must be symmetric.")


def optimize_max_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """Find portfolio weights that maximize the Sharpe Ratio.

    Returns a dict with keys: weights, return, volatility, sharpe_ratio, success, message.
    """
    _validate_inputs(expected_returns, cov_matrix)

    n = len(expected_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        ret, vol, sr = portfolio_performance(w, expected_returns, cov_matrix, risk_free_rate)
        return -sr

    result: OptimizeResult = minimize(
        neg_sharpe,
        x0=_equal_weights(n),
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed: {result.message}")

    weights = result.x
    ret, vol, sr = portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)

    return {
        "weights": pd.Series(weights, index=expected_returns.index),
        "return": ret,
        "volatility": vol,
        "sharpe_ratio": sr,
        "success": result.success,
        "message": result.message,
    }


def optimize_min_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """Find portfolio weights that minimize total variance (lowest volatility).

    Returns a dict with keys: weights, return, volatility, sharpe_ratio, success, message.
    """
    _validate_inputs(expected_returns, cov_matrix)

    n = len(expected_returns)

    def variance(w: np.ndarray) -> float:
        return portfolio_volatility(w, cov_matrix) ** 2

    result: OptimizeResult = minimize(
        variance,
        x0=_equal_weights(n),
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError(f"Min variance optimization failed: {result.message}")

    weights = result.x
    ret, vol, sr = portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)

    return {
        "weights": pd.Series(weights, index=expected_returns.index),
        "return": ret,
        "volatility": vol,
        "sharpe_ratio": sr,
        "success": result.success,
        "message": result.message,
    }


def generate_random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate random long-only portfolios and return their performance metrics.

    Returns a DataFrame with columns: return, volatility, sharpe_ratio, plus one weight
    column per ticker.
    """
    _validate_inputs(expected_returns, cov_matrix)
    if n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer.")

    rng = np.random.default_rng(seed)
    n = len(expected_returns)
    tickers = expected_returns.index.tolist()

    returns = np.empty(n_simulations)
    volatilities = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)
    weights_matrix = np.empty((n_simulations, n))

    for i in range(n_simulations):
        raw = rng.random(n)
        w = raw / raw.sum()
        ret, vol, sr = portfolio_performance(w, expected_returns, cov_matrix, risk_free_rate)
        returns[i] = ret
        volatilities[i] = vol
        sharpes[i] = sr
        weights_matrix[i] = w

    weight_cols = {ticker: weights_matrix[:, j] for j, ticker in enumerate(tickers)}

    return pd.DataFrame(
        {
            "return": returns,
            "volatility": volatilities,
            "sharpe_ratio": sharpes,
            **weight_cols,
        }
    )


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_points: int = 100,
) -> pd.DataFrame:
    """Compute the efficient frontier by minimizing volatility at each target return.

    Returns a DataFrame with columns: return, volatility, sharpe_ratio, and one weight
    column per ticker. Points where optimization failed are silently dropped.
    """
    _validate_inputs(expected_returns, cov_matrix)
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    n = len(expected_returns)
    min_ret = float(expected_returns.min())
    max_ret = float(expected_returns.max())

    if min_ret >= max_ret:
        raise ValueError(
            "All assets have the same expected return; efficient frontier is a single point."
        )

    target_returns = np.linspace(min_ret, max_ret, n_points)
    tickers = expected_returns.index.tolist()
    rows: list[dict] = []

    for target in target_returns:
        constraints = _base_constraints(n) + [
            {
                "type": "eq",
                "fun": lambda w, t=target: portfolio_return(w, expected_returns) - t,
            }
        ]

        result: OptimizeResult = minimize(
            lambda w: portfolio_volatility(w, cov_matrix) ** 2,
            x0=_equal_weights(n),
            method="SLSQP",
            bounds=_bounds(n),
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            continue

        w = result.x
        ret, vol, sr = portfolio_performance(w, expected_returns, cov_matrix, risk_free_rate)
        row: dict = {"return": ret, "volatility": vol, "sharpe_ratio": sr}
        for j, ticker in enumerate(tickers):
            row[ticker] = w[j]
        rows.append(row)

    if not rows:
        raise ValueError("Efficient frontier computation failed for all target return levels.")

    return pd.DataFrame(rows)
