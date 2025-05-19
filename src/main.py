# main.py

from data_preprocessing import load_data, preprocess_data
from model_training import evaluate_model

# Sklearn Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load and preprocess data
df = load_data("data/laptop_data.csv")      
X, y = preprocess_data(df)

# Evaluate different models
print("Evaluating Linear Regression...")
r2, mae = evaluate_model(X, y, LinearRegression())
print("Linear Regression -> R2:", r2, "MAE:", mae)

print("Evaluating Random Forest...")
r2, mae = evaluate_model(X, y, RandomForestRegressor(n_estimators=100, random_state=2))
print("Random Forest -> R2:", r2, "MAE:", mae)

print("Evaluating SVR...")
r2, mae = evaluate_model(X, y, SVR())
print("SVR -> R2:", r2, "MAE:", mae)
