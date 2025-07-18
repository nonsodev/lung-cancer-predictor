import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from dataclasses import dataclass
from src.app_logging.logger import logging
from src.exception import CustomException
import datetime

from tensorflow import keras
import tensorflow as tf
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model", "model.keras")
    trained_model_logs = os.path.join("artifacts", "model", "logs", "fit_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
class ModelTrainer:
    def __init__(self):
        self.modelConfig = ModelTrainerConfig()
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.modelConfig.trained_model_file_path), exist_ok=True)
        os.makedirs(self.modelConfig.trained_model_logs, exist_ok=True)
    
    def initiate_model_trainer(self, train_arr, test_arr, epochs=100, batch_size=32):
        try:
            logging.info("Starting model training process")
            
            # Split features and target
            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            
            # Log data info
            logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            logging.info(f"Training class distribution: {Counter(y_train)}")
            logging.info(f"Test class distribution: {Counter(y_test)}")
            
            # Create callbacks
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.modelConfig.trained_model_logs, 
                histogram_freq=1
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Build binary classification model
            model = keras.Sequential([
                keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid")  # Binary output
            ])
            
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            
            logging.info("Binary classification model created successfully")
            model.summary()
            
            # Train the model
            logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[tensorboard_callback, early_stopping],
                verbose=1
            )
            
            # Evaluate the model
            logging.info("Evaluating model on test data")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Make predictions for binary classification
            preds_proba = model.predict(X_test)
            preds = (preds_proba > 0.5).astype(int).flatten()
            
            # Generate classification report
            report = classification_report(y_test, preds)
            logging.info(f"Classification Report:\n{report}")
            
            # Additional metrics
            accuracy = accuracy_score(y_test, preds)
            conf_matrix = confusion_matrix(y_test, preds)
            
            logging.info(f"Final Accuracy: {accuracy:.4f}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
            
            # Save the model
            model.save(self.modelConfig.trained_model_file_path)
            logging.info(f"Model saved to: {self.modelConfig.trained_model_file_path}")
            
            # Return comprehensive results
            results = {
                'model': model,
                'history': history,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'predictions': preds,
                'prediction_probabilities': preds_proba
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)