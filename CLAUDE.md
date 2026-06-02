# CLAUDE.md

# Markowitz Portfolio Optimizer

Python CLI tool for portfolio risk analysis and asset allocation optimization using Modern Portfolio Theory.

## Stack

* Python 3.10+
* Libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `yfinance`
* CLI: `argparse`

## Project Structure

```text
.
├── data_loader.py
├── metrics.py
├── optimizer.py
├── main.py
├── outputs/
├── requirements.txt
└── README.md
```

## Module Responsibilities

### `data_loader.py`

* Download historical adjusted prices using `yfinance`.
* Return clean `pd.DataFrame` objects.
* Handle invalid tickers, missing data, empty downloads and date errors.

### `metrics.py`

* Calculate logarithmic returns.
* Calculate annualized expected returns using 252 trading days.
* Calculate annualized covariance matrix.
* Calculate portfolio return, volatility and Sharpe Ratio.
* Use `pd.DataFrame`, `pd.Series` and `np.ndarray` explicitly.

### `optimizer.py`

* Use `scipy.optimize.minimize` with `SLSQP`.
* Implement maximum Sharpe Ratio portfolio.
* Implement minimum variance portfolio.
* Implement random portfolio simulation.
* Implement efficient frontier calculation.
* Default constraints:

  * long-only portfolio
  * weights between 0 and 1
  * sum of weights equals 1

### `main.py`

* Build a clean CLI using `argparse`.
* Accept tickers, start date, end date, risk-free rate and number of simulations.
* Run the full workflow:

  1. download data
  2. calculate metrics
  3. optimize portfolios
  4. generate plots
  5. print results

## Visualization Rules

* Use Matplotlib only.
* Save all plots as `.png` files inside `outputs/`.
* Never open GUI windows.
* Use `plt.savefig(...)` and `plt.close()`.
* Create `outputs/` automatically if it does not exist.

## Code Quality Rules

* Use type hints.
* Use short docstrings.
* Keep functions small and focused.
* Do not mix module responsibilities.
* Do not hardcode tickers, dates or portfolio weights.
* Raise clear errors for invalid inputs.
* Prioritize readable, modular code suitable for GitHub/LinkedIn.

## Example CLI Usage

```bash
python main.py --tickers AAPL MSFT NVDA --start 2020-01-01 --end 2025-01-01 --risk-free-rate 0.02 --simulations 10000
```
