import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from dataclasses import dataclass
from src.app_logging.logger import logging
from src.exception import CustomException
import datetime

from tensorflow import keras
import tensorflow as tf

import numpy as np
from sklearn.metrics import classification_report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model", "model.keras")
    trained_model_logs = os.path.join("artifacts","model", "logs","fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
    
class ModelTrainer:
    def __init__(self):
        self.modelConfig = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1],test_arr[:,:-1], test_arr[:,-1])
            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.modelConfig.trained_model_logs, histogram_freq=1)
            model = keras.Sequential([
                keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(8,activation="relu"),
                keras.layers.Dense(4, activation="softmax"),
            ])
            
            model.compile(optimizer="SGD", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["sparse_categorical_accuracy"])
            
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
            
            preds = model.predict(X_test)
            preds = np.argmax(preds, axis=1)
            
            report = classification_report(y_test, preds)
            model.save(self.modelConfig.trained_model_file_path)
            
            return report
        except Exception as e:
            raise CustomException(e, sys)
        
    