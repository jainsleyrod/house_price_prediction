import sys
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function returns the preprocessor object which can be used to transform the data
        '''
        numeric_features = ['area','Year']
        categorical_features = ['floorRange', 'typeOfSale', 'propertyType','typeOfArea', 'mktSegment', 'region']

        num_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ]
        )
        cat_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers = [
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features),
            ]
        )

        return preprocessor
    
    def initiate_data_transformation(self,train_path,test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
    
        preprocessor_obj = self.get_data_transformer_object()

        input_feature_train_df = train_df.drop(columns=['price'],axis=1)
        target_feature_train_df = train_df['price']
        input_feature_test_df = test_df.drop(columns=['price'],axis=1)
        target_feature_test_df = test_df['price']

        #toarray
        input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df).toarray()
        input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df).toarray()

        #important step to get the transformed columns
        transformed_columns = (
            input_feature_train_df.columns[:2].tolist() +  # Numeric features
            preprocessor_obj.named_transformers_['cat'].get_feature_names_out().tolist()  # Categorical features
        )

        #converting back to df
        train_df = pd.DataFrame(input_feature_train_arr, columns=transformed_columns)
        test_df = pd.DataFrame(input_feature_test_arr, columns=transformed_columns)
        
        #adding target feature
        train_df = pd.concat([train_df, target_feature_train_df.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_df, target_feature_test_df.reset_index(drop=True)], axis=1)

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
        )

        return(
            train_df,
            test_df,
            self.data_transformation_config.preprocessor_obj_file_path
        )

        
        
