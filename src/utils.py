import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import dill



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


def check_data_corruption(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in DataFrame:")
    print(missing_values)
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print("\nDuplicate rows in DataFrame:")
    print(duplicates)
    
    # Check for inconsistent data types
    data_types = df.dtypes
    print("\nData types of DataFrame columns:")
    print(data_types)
    
    # Summary of DataFrame
    print("\nSummary of DataFrame:")
    print(df.info())
    
    return missing_values, duplicates, data_types