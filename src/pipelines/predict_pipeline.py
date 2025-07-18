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
    def __init__(self, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, 
                 allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain):
        self.yellow_fingers = yellow_fingers
        self.anxiety = anxiety
        self.peer_pressure = peer_pressure
        self.chronic_disease = chronic_disease
        self.fatigue = fatigue
        self.allergy = allergy
        self.wheezing = wheezing
        self.alcohol_consuming = alcohol_consuming
        self.coughing = coughing
        self.swallowing_difficulty = swallowing_difficulty
        self.chest_pain = chest_pain
        
    def to_dataframe(self):
        # Only include columns that are actually used by the model
        columns = ["YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE", 
                  "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
                  "SWALLOWING DIFFICULTY", "CHEST PAIN"]
        
        df = pd.DataFrame([[self.yellow_fingers, self.anxiety, self.peer_pressure, 
                           self.chronic_disease, self.fatigue, self.allergy, self.wheezing, 
                           self.alcohol_consuming, self.coughing, self.swallowing_difficulty, 
                           self.chest_pain]], columns=columns)
        
        return df

class PredictPipeline:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model(os.path.join("artifacts", "model", "model.keras"))
            self.le = load_pkl_file(os.path.join("artifacts", "le.pkl"))
            logging.info("Loaded model and label encoder successfully")
        except Exception as e:
            logging.error(f"Error loading model artifacts: {str(e)}")
            raise CustomException(e, sys)
    
    def preprocess_input(self, df):
        """Apply the same preprocessing as training"""
        try:
            # Apply label encoding to categorical columns (same as in training)
            categorical_columns = ['YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                                 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                                 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
            
            # Note: In production, you should save and load the label encoders used during training
            # For now, assuming binary encoding (0/1 or YES/NO -> 1/0)
            for col in categorical_columns:
                if col in df.columns:
                    # Convert string values to binary if needed
                    if df[col].dtype == 'object':
                        df[col] = df[col].map({'YES': 1, 'NO': 0, 'Y': 1, 'N': 0, 
                                              'True': 1, 'False': 0, 'M': 1, 'F': 0})
            
            # Create feature engineering (same as training)
            df['ANXYELFIN'] = df['ANXIETY'] * df['YELLOW_FINGERS']
            
            # No need to drop columns - they're already excluded from input
            
            logging.info(f"Final dataframe shape: {df.shape}")
            logging.info(f"Final columns: {df.columns.tolist()}")
            
            # Convert to numpy array
            return df.to_numpy()
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, 
                allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain):
        try:
            # Create data object (only with features that are actually used)
            data = CustomData(yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, 
                            allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain)
            
            df = data.to_dataframe()
            logging.info(f"Input dataframe shape: {df.shape}")
            
            # Preprocess the data
            preprocessed_data = self.preprocess_input(df)
            logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")
            
            # Make prediction (binary classification)
            pred_proba = self.model.predict(preprocessed_data, verbose=0)
            confidence = float(pred_proba[0][0])  # Probability of positive class
            
            # Convert to binary prediction
            prediction = 1 if confidence > 0.5 else 0
            
            # Convert back to meaningful labels
            prediction_label = self.le.inverse_transform([prediction])[0]
            
            logging.info("Prediction completed successfully")
            
            return {
                "prediction": prediction_label,
                "confidence": confidence,
                "risk_level": "High" if confidence > 0.7 else "Medium" if confidence > 0.3 else "Low"
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = PredictPipeline()
        
        # Example usage for lung cancer prediction (only features that are actually used)
        prediction = pipeline.predict(
            yellow_fingers="YES",
            anxiety="NO",
            peer_pressure="NO",
            chronic_disease="YES",
            fatigue="YES",
            allergy="NO",
            wheezing="YES",
            alcohol_consuming="YES",
            coughing="YES",
            swallowing_difficulty="NO",
            chest_pain="YES"
        )
        
        print(f"Lung Cancer Prediction: {prediction}")
        
    except Exception as e:
        print(f"Error: {str(e)}")