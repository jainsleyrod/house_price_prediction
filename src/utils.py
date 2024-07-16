import sqlite3
import pandas as pd


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
