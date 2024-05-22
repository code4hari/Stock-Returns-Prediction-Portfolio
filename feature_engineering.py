# feature_engineering.py
import pandas as pd

def create_features(data):
    data['Return'] = data['close'].pct_change()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['SMA_200'] = data['close'].rolling(window=200).mean()
    data['Momentum'] = data['close'] / data['close'].shift(20) - 1
    data = data.dropna()
    return data

if __name__ == "__main__":
    from data_loading import load_data
    data = load_data()
    data = create_features(data)
    print(data.head())
