import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.exception import CustomException


import pickle

def load_pkl_file(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            raise CustomException("file not exist", sys)
    except Exception as e:
        raise CustomException(e, sys)
    
    
def save_pkl_file(file_path, object):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)
    except Exception as e:
        raise CustomException(e, sys)