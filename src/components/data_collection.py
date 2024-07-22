import requests
import pandas as pd
import sqlite3


access_key = ###
access_token = ###
url = 'https://www.ura.gov.sg/uraDataService/invokeUraDS'

conn = sqlite3.connect('house_data.db')
df_list = []

for batch in range(1,4):
    # Parameters
    params = {
        'service': 'PMI_Resi_Transaction',
        'batch': batch  # Adjust batch number as needed (1-4)
    }

    # Headers
    headers = {
        'User-Agent': 'Mozilla/5.0',
        #'Accept': 'application/json',
        'AccessKey': access_key,
        'Token': access_token
    }

    # Make the request
    response = requests.get(url, headers=headers, params=params).json()

    for i in range(len(response['Result'])):
        #from the list of transactions, the last transaction is the most recent one
        variables = response['Result'][i]['transaction'][-1]
        #mkt segment column
        variables['mktSegment'] = response['Result'][i]['marketSegment']
        df_list.append(pd.DataFrame([variables]))

df = pd.concat(df_list, ignore_index=True)
if 'nettPrice' in df.columns:
        df = df.drop(columns=['nettPrice'])
df.to_sql('house_data', conn, if_exists= 'append', index= False)
conn.close()