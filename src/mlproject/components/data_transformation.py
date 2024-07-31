import sys
from dataclasses import dataclass
import  numpy as np 
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
import os


@dataclass

class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    # class implementation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig


    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            
            ]            

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scalar', StandardScaler(with_mean=False))

            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("car_pipeline",cat_pipeline,categorical_columns)


            
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Reading the train and test ffile")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["reading_score", "writing_score"]

            #divide the train dataset into independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            #divide the test dataset into independent and dependent feature

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing on training and test data")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

        
            )





        except Exception as e:
            raise CustomException(sys,e)        




