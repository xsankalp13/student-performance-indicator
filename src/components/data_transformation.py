import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsible for data tranformation
        '''
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", 
                                    "race_ethnicity",
                                    "parental_level_of_education", 
                                    "lunch", 
                                    "test_preparation_course"]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )    

            logging.info("Numerical columns standardization completed")

            categorical_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder() ),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")



            logging.info(f"categorical columns: {categorical_features}")
            logging.info(f"numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer([
                ("numerical_pipline", numerical_pipeline, numerical_features),
                ("categorical_pipline", categorical_pipline, categorical_features)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessor object")

            preprocessing_object = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_features = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            logging.info(f"Applying preprocessor object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)
