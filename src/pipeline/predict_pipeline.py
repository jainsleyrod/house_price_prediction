import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path = 'artifacts/model.pkl'
        preprocess_path = 'artifacts/preprocessor.pkl'
        model = load_object(model_path)
        preprocessor = load_object(preprocess_path)
        scaled_features = preprocessor.transform(features)
        pred = model.predict(scaled_features)
        return pred


class CustomData:
    def __init__(self,
        area: int,
        typeOfSale: str,
        region: str,
        floorRange: str,
        Year: int,
        mktSegment: str,
        propertyType: str,
        typeOfArea: str):

        self.area = area
        self.typeOfSale = typeOfSale
        self.region = region
        self.floorRange = floorRange
        self.Year = Year
        self.mktSegment = mktSegment,
        self.propertyType = propertyType,
        self.typeOfArea = typeOfArea
    
    def get_data_as_df(self):
        data = {
            'area': [self.area],
            'typeOfSale': [self.typeOfSale],
            'region': [self.region],
            'floorRange': [self.floorRange],
            'Year': [self.Year],
            'mktSegment': [self.mktSegment],
            'propertyType': [self.propertyType],
            'typeOfArea': [self.typeOfArea]
        }
        return pd.DataFrame(data)
    