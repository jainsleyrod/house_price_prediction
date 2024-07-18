import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import retrieve_data_from_db
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


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

        #covert features to numerical
        df['area'] = pd.to_numeric(df['area'])
        df['price'] = pd.to_numeric(df['price'])
        
        #remove any whitspace in column name
        df.columns = df.columns.str.replace(r'\s+', '', regex=True)

        #remove some outliers
        area_threshold = df['area'].quantile(0.99)
        price_threshold = df['price'].quantile(0.99)
        df = df[(df['area'] <= area_threshold) & (df['price'] <= price_threshold)]
        
        #creating new columns, dropping unwanted columns
        df['Year'] = "20" + df['contractDate'].astype(str).str[-2:]
        df['Year'] = pd.to_numeric(df['Year'])
        
        district_to_region = {
            '01': 'Central', '02': 'Central', '03': 'Central', '04': 'Central', '05': 'West', '06': 'Central',
            '07': 'Central', '08': 'Central', '09': 'Central', '10': 'Central', '11': 'Central',
            '12': 'East', '13': 'East', '14': 'East', '15': 'East', '16': 'East', '17': 'East', '18': 'East',
            '19': 'North-East', '20': 'North-East', '21': 'Central', '22': 'West', '23': 'West', '24': 'West',
            '25': 'North', '26': 'North', '27': 'North', '28': 'North-East'
        }
        
        df['region'] = df['district'].map(district_to_region)
        df = df.drop(['tenure','noOfUnits','contractDate','district'],axis=1)
        
        df['floorRange'] = df['floorRange'].apply(lambda x: 'Ground' if x == '-' else x)
        df['typeOfSale'] = df['typeOfSale'].map({ '1' : 'New Sale', '2': 'Sub Sale', '3': 'Resale'})
        
    
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
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_df,test_df,_ = data_transformation_obj.initiate_data_transformation(train_path,test_path)