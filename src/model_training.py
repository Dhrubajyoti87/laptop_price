from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.data_preprocessing import load_data, preprocess_data

# Load the raw CSV data
df = load_data("data/laptop_data.csv")

# Preprocess the data
X, y = preprocess_data(df)


def evaluate_model(model):
    column_transformer=ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
    ],remainder='passthrough')
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.15,random_state=2)
    pipe = Pipeline([
        ('transform', column_transformer),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return r2, mae



