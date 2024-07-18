import os
from dataclasses import dataclass

#you should try every model available in the library and compare the results
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_df, test_df):
        X_train = train_df.drop(columns=['price'], axis=1)
        y_train = train_df['price']
        X_test = test_df.drop(columns=['price'], axis=1)
        y_test = test_df['price']
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor(),
            'KNN': KNeighborsRegressor(),
            'CatBoost': CatBoostRegressor(verbose=0)
        }
        #params for each model for hyperparamter tuning
        params = {
            'Linear Regression': {},
            'Decision Tree': {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            'Random Forest': {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            'Gradient Boosting': {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            'XGBoost': {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            'CatBoost': {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            'AdaBoost': {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7]
            }
        }

        
        report = evaluate_models(X_train, y_train, X_test, y_test, models,params)
        best_model_score = max(sorted(report.values()))
        best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
        best_model = models[best_model_name]
        
        save_object(self.model_trainer_config.trained_model_file_path,
                    best_model)
        print(f"Best Model: {best_model_name}")
        predict_test = best_model.predict(X_test)
        r2 = r2_score(y_test, predict_test)
        print(f"R2 Score: {r2}")

        return self.model_trainer_config.trained_model_file_path
