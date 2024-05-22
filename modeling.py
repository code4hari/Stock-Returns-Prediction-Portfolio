# modeling.py
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def train_models(X_train, y_train):
    models = {
        'ridge': Ridge(),
        'gbr': GradientBoostingRegressor(),
        'nn': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f'{name} model trained.')
    
    return models

def evaluate_models(models, X_test, y_test):
    predictions = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        predictions[name] = preds
        mse = mean_squared_error(y_test, preds)
        print(f'{name} model MSE: {mse}')
    
    return predictions

if __name__ == "__main__":
    from feature_engineering import create_features
    from data_loading import load_data
    from sklearn.model_selection import train_test_split

    data = load_data()
    data = create_features(data)
    
    features = ['SMA_50', 'SMA_200', 'Momentum']
    X = data[features]
    y = data['Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = train_models(X_train, y_train)
    predictions = evaluate_models(models, X_test, y_test)
    print(predictions)
