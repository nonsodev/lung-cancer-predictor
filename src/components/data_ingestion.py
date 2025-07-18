import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from src.exception import CustomException
from src.app_logging.logger import logging



from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join("artifacts","train.csv")
    test_data_path :str = os.path.join("artifacts","test.csv")
    raw_data_path :str = os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            data_path = os.path.join("notebooks","survey_lung_cancer.csv")
            df=pd.read_csv(data_path)
            logging.info("read the dataset into a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=None, header=True)

            logging.info("initiating train test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of data complete")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
 
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
    transformer = DataTransformation(columns_to_drop=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
    train_arr, test_arr = transformer.preprocess_data(train_data_path, test_data_path)
    print(train_arr, test_arr)
    trainer = ModelTrainer()
    report = trainer.initiate_model_trainer(train_arr, test_arr)
    print(report)
    
    
    
        