"""Download and clean historical price data using yfinance."""

import pandas as pd
import yfinance as yf


def download_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers and date range.

    Returns a DataFrame with dates as index and tickers as columns.
    """
    _validate_inputs(tickers, start, end)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}. Check tickers and date range.")

    prices: pd.DataFrame = raw["Close"]

    # yfinance returns a Series when a single string ticker is passed;
    # normalise to DataFrame so the rest of the pipeline is always consistent.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")

    if prices.empty:
        raise ValueError(f"All rows were NaN after cleaning for tickers: {tickers}.")

    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise ValueError(f"Tickers with no valid price data: {missing}.")

    return prices


def _validate_inputs(tickers: list[str], start: str, end: str) -> None:
    """Raise ValueError for empty tickers list or unparseable dates."""
    if not tickers:
        raise ValueError("Tickers list cannot be empty.")

    if not start:
        raise ValueError("Start date must be provided.")

    if not end:
        raise ValueError("End date must be provided.")

    try:
        pd.to_datetime(start)
    except Exception:
        raise ValueError(f"Invalid start date: {start!r}. Use YYYY-MM-DD format.")

    try:
        pd.to_datetime(end)
    except Exception:
        raise ValueError(f"Invalid end date: {end!r}. Use YYYY-MM-DD format.")
