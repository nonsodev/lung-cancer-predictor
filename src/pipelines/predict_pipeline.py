import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.app_logging.logger import logging
from src.exception import CustomException
from src.utils.utils import load_pkl_file

import pandas as pd
import numpy as np
import tensorflow as tf

class CustomData:
    def __init__(self,Temperature,Humidity, Wind_Speed, Precipitation, Cloud_cover, Atmospheric_pressure, UV_Index, Season, Visibility, Location):
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.Wind = Wind_Speed
        self.Precipitation = Precipitation
        self.Cloud_cover = Cloud_cover
        self.Atmospheric_pressure = Atmospheric_pressure
        self.UV_Index = UV_Index
        self.Season = Season
        self.Visibility = Visibility
        self.Location = Location
    def to_dataframe(self):
        columns = ["Temperature","Humidity","Wind Speed","Precipitation (%)","Cloud Cover","Atmospheric Pressure","UV Index","Season","Visibility (km)","Location"]
        
        df = pd.DataFrame([[self.Temperature, self.Humidity, self.Wind, self.Precipitation, self.Cloud_cover, self.Atmospheric_pressure, self.UV_Index, self.Season, self.Visibility, self.Location]],
                          columns=columns)
        
        return df
    
class PredictPipeline:
    def __init__(self):
        self.model = tf.keras.models.load_model(os.path.join("artifacts","model","model.keras"))
        self.le = load_pkl_file(os.path.join("artifacts","le.pkl"))
        self.preprocessor = load_pkl_file(os.path.join("artifacts","preprocessor.pkl"))
        logging.info("loaded in prerequisites")
    def predict(self,Temperature,Humidity, Wind_Speed, Precipitation, Cloud_cover, Atmospheric_pressure, UV_Index, Season, Visibility, Location):
        data = CustomData(Temperature,Humidity, Wind_Speed, Precipitation, Cloud_cover, Atmospheric_pressure, UV_Index, Season, Visibility, Location)
        df = data.to_dataframe()
        preprocessed_data = self.preprocessor.transform(df)
        pred_arr = self.model.predict(preprocessed_data[0:1])
        pred_ind = np.argmax(pred_arr, axis=1)
        score = pred_arr[0,pred_ind[0]]
        pred_weather_type = self.le.inverse_transform(pred_ind)[0]
        
        logging.info("predicted and returning results")
        return {"weather_type":pred_weather_type, "confidence": score}
    
if __name__ == "__main__":
    pipeline = PredictPipeline()
    prediction = pipeline.predict(32.0,66,6.0,92.0,"partly cloudy",1010.21,2,"Spring",1.5,"inland")
    print(prediction)
    
        
        
        
        
        
        