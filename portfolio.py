# portfolio.py
import cvxpy as cp
import numpy as np
import pandas as pd

def mean_variance_optimization(returns):
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)

    n = returns.shape[1]
    mu = returns.mean(axis=0).values.reshape(-1, 1)  # Convert to numpy array and reshape
    Sigma = returns.cov().values

    w = cp.Variable(n)
    risk_aversion = 1.0  # Adjust as needed
    objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value

if __name__ == "__main__":
    # Example usage
    returns = pd.DataFrame(np.random.randn(100, 5))  # Example returns data
    weights = mean_variance_optimization(returns)
    print(weights)
