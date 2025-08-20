import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransform:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer_object(self):

        try:
            numerical_columns = ['rate','area_sqft','bedRoom','bathroom','balcony','noOfFloor','agePossession_processed','avg_rating']
            categorical_columns = ['facing']

            num_pipline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            logging.info('Numerical columns standard Scaling Completed')
            logging.info('Catgorical cloumns onehotencoding completed.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipline',num_pipline,numerical_columns),
                    ('cat_pipline',cat_pipline,categorical_columns)
                ]
            )

            return preprocessor
        

        except Exception as err:
            raise CustomException(err, sys)
    

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read train and test data completed.')
            logging.info('obtainig preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_cloumn_name = 'price'
            features_columns = ['facing','rate','area_sqft','bedRoom','bathroom','balcony','noOfFloor',
                    'agePossession_processed','avg_rating']
            
            input_feature_train_df = train_df[features_columns]
            target_feature_train_df = train_df[target_cloumn_name]

            input_feature_test_df = test_df[features_columns]
            target_test_df = test_df[target_cloumn_name]


            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]

            logging.info('Saved preprocessing object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as err:
            raise CustomException(err, sys) 