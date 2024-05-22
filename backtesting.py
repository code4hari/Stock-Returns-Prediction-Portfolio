# backtesting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_portfolio(weights, returns):
    portfolio_returns = returns @ weights
    cum_returns = (1 + portfolio_returns).cumprod()
    return cum_returns

def plot_performance(cum_returns, benchmark):
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns, label='Portfolio')
    plt.plot(benchmark, label='Benchmark')
    plt.title('Growth of a Dollar')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    returns = pd.DataFrame(np.random.randn(100, 1))  # Example returns data
    weights = np.array([1])  # Example weights
    cum_returns = backtest_portfolio(weights, returns)
    benchmark = pd.Series(np.random.randn(100).cumsum())  # Example benchmark
    plot_performance(cum_returns, benchmark)
