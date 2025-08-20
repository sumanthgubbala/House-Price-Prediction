import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransform



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data Ingestion method')

        try:
            df = pd.read_csv(r'notebook\data\houes_cleaned_data.csv')
            logging.info('Read the dataset as DataFrame')

            selected_columns = ['facing','rate','area_sqft','bedRoom','bathroom','balcony','noOfFloor',
                    'agePossession_processed','avg_rating','price']
            
            df = df[selected_columns]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated ')
            train_set, test_set = train_test_split(df, test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as err:
            logging.info("EXception occured at Data Ingestion Stage ")
            raise CustomException(err, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()
    print(train_data, test_data)

    data_transformation = DataTransform()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    print(train_arr, test_arr)


