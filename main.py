"""Entry point: parse CLI arguments and run the portfolio optimization workflow."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import download_price_data
from metrics import calculate_log_returns, annualize_returns, annualize_covariance, benchmark_metrics
from optimizer import (
    optimize_max_sharpe,
    optimize_min_variance,
    generate_random_portfolios,
    efficient_frontier,
)


def build_parser() -> argparse.ArgumentParser:
    """Define and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Markowitz Portfolio Optimizer"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Space-separated list of ticker symbols (e.g. AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate as a decimal (default: 0.0)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10_000,
        help="Number of random portfolio simulations (default: 10000)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark ticker to compare against (default: SPY)",
    )
    return parser


def _print_portfolio(label: str, result: dict) -> None:
    """Print portfolio metrics and weights to the terminal."""
    print(f"\n{'─' * 42}")
    print(f"  {label}")
    print(f"{'─' * 42}")
    print(f"  Return:     {result['return']:>8.2%}")
    print(f"  Volatility: {result['volatility']:>8.2%}")
    print(f"  Sharpe:     {result['sharpe_ratio']:>8.4f}")
    print(f"\n  Weights:")
    for ticker, weight in result["weights"].items():
        print(f"    {ticker:<10} {weight:>7.2%}")


def _print_benchmark(bm: dict) -> None:
    """Print benchmark metrics to the terminal."""
    print(f"\n{'─' * 42}")
    print(f"  Benchmark: {bm['ticker']}")
    print(f"{'─' * 42}")
    print(f"  Return:     {bm['return']:>8.2%}")
    print(f"  Volatility: {bm['volatility']:>8.2%}")
    print(f"  Sharpe:     {bm['sharpe_ratio']:>8.4f}")


def _save_plot(
    random_portfolios: pd.DataFrame,
    frontier: pd.DataFrame,
    max_sharpe: dict,
    min_var: dict,
    output_path: str,
    benchmark: dict | None = None,
) -> None:
    """Render efficient frontier chart and save to disk without opening a window."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        random_portfolios["volatility"],
        random_portfolios["return"],
        c=random_portfolios["sharpe_ratio"],
        cmap="viridis",
        alpha=0.4,
        s=5,
        label="Random portfolios",
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    ax.plot(
        frontier["volatility"],
        frontier["return"],
        color="black",
        linewidth=2,
        label="Efficient frontier",
    )

    ax.scatter(
        max_sharpe["volatility"],
        max_sharpe["return"],
        marker="*",
        color="gold",
        edgecolors="black",
        s=300,
        zorder=5,
        label=f"Max Sharpe ({max_sharpe['sharpe_ratio']:.2f})",
    )

    ax.scatter(
        min_var["volatility"],
        min_var["return"],
        marker="D",
        color="red",
        edgecolors="black",
        s=100,
        zorder=5,
        label="Min Variance",
    )

    if benchmark is not None:
        ax.scatter(
            benchmark["volatility"],
            benchmark["return"],
            marker="s",
            color="cyan",
            edgecolors="black",
            s=150,
            zorder=5,
            label=f"{benchmark['ticker']} (SR={benchmark['sharpe_ratio']:.2f})",
        )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier — Markowitz Portfolio Optimization")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run the full portfolio optimization workflow."""
    parser = build_parser()
    args = parser.parse_args()

    print(f"Tickers:   {', '.join(args.tickers)}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Period:    {args.start} → {args.end}")
    print(f"Rf rate:   {args.risk_free_rate:.2%}  |  Simulations: {args.simulations:,}")

    print("\nDownloading price data...")
    try:
        prices = download_price_data(args.tickers, args.start, args.end)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    log_returns = calculate_log_returns(prices)
    expected_returns = annualize_returns(log_returns)
    cov_matrix = annualize_covariance(log_returns)

    print(f"Downloading benchmark data ({args.benchmark})...")
    benchmark: dict | None = None
    try:
        benchmark_prices = download_price_data([args.benchmark], args.start, args.end)
        b_ret, b_vol, b_sr = benchmark_metrics(benchmark_prices, args.risk_free_rate)
        benchmark = {
            "ticker": args.benchmark,
            "return": b_ret,
            "volatility": b_vol,
            "sharpe_ratio": b_sr,
        }
    except ValueError as exc:
        print(f"Warning: Could not load benchmark data: {exc}", file=sys.stderr)

    print("Running optimizations...")
    try:
        max_sharpe = optimize_max_sharpe(expected_returns, cov_matrix, args.risk_free_rate)
        min_var = optimize_min_variance(expected_returns, cov_matrix, args.risk_free_rate)
        random_portfolios = generate_random_portfolios(
            expected_returns, cov_matrix, args.risk_free_rate, args.simulations
        )
        frontier = efficient_frontier(expected_returns, cov_matrix, args.risk_free_rate)
    except ValueError as exc:
        print(f"Optimization error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe)
    _print_portfolio("Minimum Variance Portfolio", min_var)
    if benchmark is not None:
        _print_benchmark(benchmark)

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "efficient_frontier.png")
    _save_plot(random_portfolios, frontier, max_sharpe, min_var, output_path, benchmark)
    print(f"\nChart saved → {output_path}")


if __name__ == "__main__":
    main()
