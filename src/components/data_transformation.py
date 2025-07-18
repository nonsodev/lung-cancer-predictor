import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from collections import Counter

from src.exception import CustomException
from src.app_logging.logger import logging
from src.utils.utils import save_pkl_file, load_pkl_file


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join("artifacts", "le.pkl")

class DataTransformation:
    def __init__(self, 
                 columns_to_drop=None, 
                 oversampling_strategy='SMOTE',
                 oversampling_ratio='auto'):
        """
        Initialize DataTransformation with optional parameters
        
        Args:
            columns_to_drop (list): List of column names to drop from the dataset
            oversampling_strategy (str): 'SMOTE', 'RandomOverSampler', 'ADASYN', or None
            oversampling_ratio (str/dict): 'auto', 'minority', 'not minority', 'all' or dict
        """
        self.data_transformation_config = DataTransformationConfig()
        self.columns_to_drop = columns_to_drop or []
        self.oversampling_strategy = oversampling_strategy
        self.oversampling_ratio = oversampling_ratio
        self.preprocessor = self.get_data_transformer_object()
        self.oversampler = self.get_oversampler()
    
    def get_oversampler(self):
        """Get the oversampling object based on strategy"""
        try:
            if self.oversampling_strategy == 'SMOTE':
                return SMOTE(sampling_strategy=self.oversampling_ratio, random_state=42)
            elif self.oversampling_strategy == 'RandomOverSampler':
                return RandomOverSampler(sampling_strategy=self.oversampling_ratio, random_state=42)
            elif self.oversampling_strategy == 'ADASYN':
                return ADASYN(sampling_strategy=self.oversampling_ratio, random_state=42)
            else:
                return None
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_transformer_object(self):
        logging.info("getting preprocessor")
        try:
            # Define all possible columns
            all_numerical_cols = []
            all_categorical_cols = ["Cloud Cover", "Season", "Location"]
            
            # Remove columns that should be dropped
            numerical_cols = [col for col in all_numerical_cols if col not in self.columns_to_drop]
            categorical_cols = [col for col in all_categorical_cols if col not in self.columns_to_drop]
            
            logging.info(f"Using numerical columns: {numerical_cols}")
            logging.info(f"Using categorical columns: {categorical_cols}")
            logging.info(f"Dropped columns: {self.columns_to_drop}")
            
            # Create pipelines
            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # Added imputer for safety
                ("scaler", StandardScaler(with_mean=False))
            ])
            
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Added imputer for safety
                ("encoder", OneHotEncoder(handle_unknown='ignore'))  # Added handle_unknown for safety
            ])
            
            # Create preprocessor only with non-empty column lists
            transformers = []
            if numerical_cols:
                transformers.append(("numerical_pipeline", numerical_pipeline, numerical_cols))
            if categorical_cols:
                transformers.append(("categorical_pipeline", categorical_pipeline, categorical_cols))
            
            if not transformers:
                raise ValueError("No columns left after dropping specified columns")
            
            preprocessor = ColumnTransformer(transformers)
            logging.info("saved and returning preprocessor")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_data(self, train_path, test_path):
        try:
            logging.info("began preprocessing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Original train shape: {train_df.shape}")
            logging.info(f"Original test shape: {test_df.shape}")
            
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()
            
            le = LabelEncoder()
            
            train_df['GENDER']=le.fit_transform(train_df['GENDER'])
            train_df['LUNG_CANCER']=le.fit_transform(train_df['LUNG_CANCER'])
            train_df['SMOKING']=le.fit_transform(train_df['SMOKING'])
            train_df['YELLOW_FINGERS']=le.fit_transform(train_df['YELLOW_FINGERS'])
            train_df['ANXIETY']=le.fit_transform(train_df['ANXIETY'])
            train_df['PEER_PRESSURE']=le.fit_transform(train_df['PEER_PRESSURE'])
            train_df['CHRONIC DISEASE']=le.fit_transform(train_df['CHRONIC DISEASE'])
            train_df['FATIGUE ']=le.fit_transform(train_df['FATIGUE '])
            train_df['ALLERGY ']=le.fit_transform(train_df['ALLERGY '])
            train_df['WHEEZING']=le.fit_transform(train_df['WHEEZING'])
            train_df['ALCOHOL CONSUMING']=le.fit_transform(train_df['ALCOHOL CONSUMING'])
            train_df['COUGHING']=le.fit_transform(train_df['COUGHING'])
            train_df['SHORTNESS OF BREATH']=le.fit_transform(train_df['SHORTNESS OF BREATH'])
            train_df['SWALLOWING DIFFICULTY']=le.fit_transform(train_df['SWALLOWING DIFFICULTY'])
            train_df['CHEST PAIN']=le.fit_transform(train_df['CHEST PAIN'])
            
            test_df['GENDER']=le.fit_transform(test_df['GENDER'])
            test_df['LUNG_CANCER']=le.fit_transform(test_df['LUNG_CANCER'])
            test_df['SMOKING']=le.fit_transform(test_df['SMOKING'])
            test_df['YELLOW_FINGERS']=le.fit_transform(test_df['YELLOW_FINGERS'])
            test_df['ANXIETY']=le.fit_transform(test_df['ANXIETY'])
            test_df['PEER_PRESSURE']=le.fit_transform(test_df['PEER_PRESSURE'])
            test_df['CHRONIC DISEASE']=le.fit_transform(test_df['CHRONIC DISEASE'])
            test_df['FATIGUE ']=le.fit_transform(test_df['FATIGUE '])
            test_df['ALLERGY ']=le.fit_transform(test_df['ALLERGY '])
            test_df['WHEEZING']=le.fit_transform(test_df['WHEEZING'])
            test_df['ALCOHOL CONSUMING']=le.fit_transform(test_df['ALCOHOL CONSUMING'])
            test_df['COUGHING']=le.fit_transform(test_df['COUGHING'])
            test_df['SHORTNESS OF BREATH']=le.fit_transform(test_df['SHORTNESS OF BREATH'])
            test_df['SWALLOWING DIFFICULTY']=le.fit_transform(test_df['SWALLOWING DIFFICULTY'])
            test_df['CHEST PAIN']=le.fit_transform(test_df['CHEST PAIN'])
            
            train_df['ANXYELFIN']=train_df['ANXIETY']*train_df['YELLOW_FINGERS']
            test_df['ANXYELFIN']=test_df['ANXIETY']*test_df['YELLOW_FINGERS']
            
            
            # Drop specified columns
            if self.columns_to_drop:
                logging.info(f"Dropping columns: {self.columns_to_drop}")
                train_df = train_df.drop(columns=self.columns_to_drop, errors='ignore')
                test_df = test_df.drop(columns=self.columns_to_drop, errors='ignore')
                logging.info(f"After dropping - train shape: {train_df.shape}, test shape: {test_df.shape}")
            
            # Separate features and target
            train_input_df = train_df.drop("LUNG_CANCER", axis=1)
            test_input_df = test_df.drop("LUNG_CANCER", axis=1)
            
            train_output_series = train_df["LUNG_CANCER"]
            test_output_series = test_df["LUNG_CANCER"]
            
            # Show class distribution before oversampling
            logging.info(f"Original class distribution: {Counter(train_output_series)}")
            
            # Encode target variable
            le = LabelEncoder()
            train_output_arr = le.fit_transform(train_output_series)
            test_output_arr = le.transform(test_output_series)
            
            # Preprocess features
            preprocessor = self.preprocessor
            train_input_arr = preprocessor.fit_transform(train_input_df)
            test_input_arr = preprocessor.transform(test_input_df)
            logging.info("splitted and preprocessed features")
            
            # Apply oversampling if specified
            if self.oversampler is not None:
                logging.info(f"Applying {self.oversampling_strategy} oversampling...")
                train_input_arr, train_output_arr = self.oversampler.fit_resample(
                    train_input_arr, train_output_arr
                )
                logging.info(f"After oversampling - train shape: {train_input_arr.shape}")
                logging.info(f"New class distribution: {Counter(train_output_arr)}")
            
            # Combine features and target
            train_arr = np.c_[train_input_arr, train_output_arr]
            test_arr = np.c_[test_input_arr, test_output_arr]
            logging.info("joined the input and output together")
            
            # Save objects (Note: oversampler is not saved as it's only used during training)
            save_pkl_file(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            save_pkl_file(self.data_transformation_config.label_encoder_obj_file_path, le)
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e, sys)


