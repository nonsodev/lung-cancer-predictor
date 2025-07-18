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
    
    
    def preprocess_data(self, train_path, test_path):
        try:
            logging.info("began preprocessing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Original train shape: {train_df.shape}")
            logging.info(f"Original test shape: {test_df.shape}")
            
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()
            
            # Define columns to encode
            categorical_columns = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                                 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
                                 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                                 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
            
            # Create label encoders for each categorical column
            label_encoders = {}
            
            # Fit and transform categorical columns for train data
            for col in categorical_columns:
                if col in train_df.columns:
                    le = LabelEncoder()
                    train_df[col] = le.fit_transform(train_df[col])
                    label_encoders[col] = le
                    
                    # Transform test data using the same encoder (only transform, not fit)
                    if col in test_df.columns:
                        test_df[col] = le.transform(test_df[col])
            
            # Create feature engineering
            train_df['ANXYELFIN'] = train_df['ANXIETY'] * train_df['YELLOW_FINGERS']
            test_df['ANXYELFIN'] = test_df['ANXIETY'] * test_df['YELLOW_FINGERS']
            
            # Drop specified columns
            if self.columns_to_drop:
                logging.info(f"Dropping columns: {self.columns_to_drop}")
                train_df = train_df.drop(columns=self.columns_to_drop, errors='ignore')
                test_df = test_df.drop(columns=self.columns_to_drop, errors='ignore')
                logging.info(f"After dropping - train shape: {train_df.shape}, test shape: {test_df.shape}")
            
            logging.info(f"final available column names: {train_df.columns}")
            
            # Separate features and target
            train_input_df = train_df.drop("LUNG_CANCER", axis=1)
            test_input_df = test_df.drop("LUNG_CANCER", axis=1)
            
            train_output_series = train_df["LUNG_CANCER"]
            test_output_series = test_df["LUNG_CANCER"]
            
            # Show class distribution before oversampling
            logging.info(f"Original class distribution: {Counter(train_output_series)}")
            
            # Get input as array
            train_input_arr = train_input_df.to_numpy()
            test_input_arr = test_input_df.to_numpy()
            
            # Encode target variable with separate encoder
            target_le = LabelEncoder()
            train_output_arr = target_le.fit_transform(train_output_series)
            test_output_arr = target_le.transform(test_output_series)
            
            logging.info("splitted and preprocessed features")
            
            # Apply oversampling ONLY to training data
            if self.oversampler is not None:
                logging.info(f"Applying {self.oversampling_strategy} oversampling to training data only...")
                train_input_arr, train_output_arr = self.oversampler.fit_resample(
                    train_input_arr, train_output_arr
                )
                logging.info(f"After oversampling - train shape: {train_input_arr.shape}")
                logging.info(f"New train class distribution: {Counter(train_output_arr)}")
                logging.info(f"Test data unchanged - test shape: {test_input_arr.shape}")
            
            # Combine features and target
            train_arr = np.c_[train_input_arr, train_output_arr]
            test_arr = np.c_[test_input_arr, test_output_arr]
            logging.info("joined the input and output together")
            
            # Save the target label encoder
            save_pkl_file(self.data_transformation_config.label_encoder_obj_file_path, target_le)
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e, sys)