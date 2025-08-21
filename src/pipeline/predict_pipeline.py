import sys
import pandas as pd
import os
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:

    def __init__(self) -> None:
        pass


    def predict(self,features):

        try:
            
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            logging.info("Loading model")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info('data transfrom into scaled data ')
            data_scaled = preprocessor.transform(features)
            logging.info("Making prediction")
            prediction = model.predict(data_scaled)
            return np.round(prediction)
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:

    def __init__(self,facing: str, rate: float, area_sqft: float, bedRoom: int, bathroom: int ,balcony: int,noOfFloor : int ,agePossession_processed:float ,avg_rating: float):

        self.facing = facing
        self.rate = rate
        self.area_sqft = area_sqft
        self.bedRoom = bedRoom
        self.bathroom = bathroom
        self.balcony = balcony
        self.noOfFloor = noOfFloor
        self.agePossession_processed = agePossession_processed
        self.avg_rating = avg_rating
    

    def get_data_as_dataframe(self):

        try:
            custom_data_input ={
                'facing':[self.facing],
                'rate': [self.rate],
                'area_sqft':[self.area_sqft],
                'bedRoom':[self.bedRoom],
                'bathroom':[self.bathroom],
                'balcony':[self.balcony],
                'noOfFloor':[self.noOfFloor],
                'agePossession_processed':[self.agePossession_processed],
                'avg_rating':[self.avg_rating]
            }
            return pd.DataFrame(custom_data_input)
        
        except Exception as e :
            raise CustomException(e,sys)



# if __name__ == '__main__':

#     data = CustomData('West',10,100,1,1,0,0,0,0)
#     df = data.get_data_as_dataframe()

#     model = PredictionPipeline()
#     pred = model.predict(df)

#     print(df)
#     print(pred)