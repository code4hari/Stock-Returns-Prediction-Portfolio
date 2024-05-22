# data_loading.py
import pandas as pd
from config import DATA_PATH

def load_data():
    data = pd.read_csv(DATA_PATH)
    # Example preprocessing: parse dates and set index
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

if __name__ == "__main__":
    data = load_data()
    print(data.head())
