# main.py
import pandas as pd
from data_loading import load_data
from feature_engineering import create_features
from modeling import train_models, evaluate_models
from portfolio import mean_variance_optimization
from backtesting import backtest_portfolio, plot_performance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess data
data = load_data()
data = create_features(data)

# Prepare data for modeling
features = ['SMA_50', 'SMA_200', 'Momentum']
X = data[features]
y = data['Return']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train models and evaluate predictions
models = train_models(X_train, y_train)
predictions = evaluate_models(models, X_test, y_test)

# Backtest the portfolio
returns = pd.DataFrame(data['Return'][-len(y_test):].values, columns=['Return'])  # Ensure returns is a DataFrame
weights = mean_variance_optimization(returns)
cum_returns = backtest_portfolio(weights, returns)

# Benchmark: Assume S&P 500 (or any relevant benchmark) data is available
benchmark = data['close'][-len(y_test):].pct_change().fillna(0)
benchmark_cum_returns = (1 + benchmark).cumprod()

# Plot performance
plt.figure(figsize=(14, 7))
plt.plot(cum_returns, label='Portfolio')
plt.plot(benchmark_cum_returns, label='Benchmark')
plt.title('Portfolio vs. Benchmark Performance')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()

# Zoom in by setting y-axis limits based on min and max values
min_return = min(cum_returns.min(), benchmark_cum_returns.min())
max_return = max(cum_returns.max(), benchmark_cum_returns.max())
plt.ylim(min_return * 0.9, max_return * 1.1)  # Adjust these multipliers as needed

plt.grid(True)
plt.show()