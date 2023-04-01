## import required libraies
import os
import sys 
from gemstone.exception import CustomException
from gemstone.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Creating the class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Data ingestion method started.")
        try:
            df = pd.read_csv('/config/workspace/notebooks/data/gemstone.csv')
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()