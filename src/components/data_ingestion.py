import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import retrieve_data_from_db


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        df = retrieve_data_from_db(r'C:\Users\James\OneDrive\Desktop\house_price_prediction\house_data.db','house_data')

        #create artifacts dir
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
        df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
        train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
        test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()