import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def retrieve_data_from_db(db_path, table_name):
    """
    Retrieve data from the specified table in the SQLite database and return it as a Pandas DataFrame.

    Parameters:
        db_path (str): Full path to the SQLite database file.
        table_name (str): Name of the table to retrieve data from. Default is 'house_data'.

    Returns:
        pd.DataFrame: DataFrame containing the retrieved data.
    """
   
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def save_object(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as f:
        dill.dump(obj,f)


def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    report = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]

        param = params[list(models.keys())[i]]
        gs = GridSearchCV(model,param,cv=3)
        gs.fit(X_train,y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)


        #predicting on the training and test set
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train,y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)
        report[list(models.keys())[i]] = test_model_score  
    return report