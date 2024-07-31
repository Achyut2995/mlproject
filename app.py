from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
import sys

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
       data_ingestion = DataIngestion()
       data_ingestion.initiate_data_ingestion()
       train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

       
       data_transformation=DataTransformation()
       data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    except Exception as e:
        logging.error("Custom Exception")
        raise CustomException(e, sys)