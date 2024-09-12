import numpy as np
import pandas as pd

import pickle
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from houses.params import *

# FEATURES = ['LotArea', 'OverallCond', 'GrLivArea', 'BedroomAbvGr', 'PoolArea', \
#                'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'Neighborhood', 'CentralAir']

def preprocess(df):
    """
    This function cleans and preprocesses the input df and returns processed df
    """

    # Drop duplicates
    df = df.drop_duplicates(df)

    # Choose relevant features
    X_train_app = df[FEATURES]

    # Convert continuous variables to categorical
    X_train_app['PoolArea'][X_train_app['PoolArea'] > 0] = 1
    X_train_app['GarageArea'][X_train_app['GarageArea'] > 0] = 1
    X_train_app['BsmtFinSF1'][X_train_app['BsmtFinSF1'] > 0] = 1
    X_train_app['Fireplaces'][X_train_app['Fireplaces'] > 0] = 1

    # Creating separate lists for numeric and categorical features
    num_cols = ['LotArea', 'OverallCond', 'GrLivArea', 'BedroomAbvGr']
    cat_cols = ['PoolArea', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'Neighborhood', 'CentralAir']

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    X_train_app[num_cols] = num_imputer.fit_transform(X_train_app[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_app[cat_cols] = cat_imputer.fit_transform(X_train_app[cat_cols])

    # Scale numeric features
    scaler = MinMaxScaler()
    X_train_app[num_cols] = scaler.fit_transform(X_train_app[num_cols])

    # Encode object categorical features
    X_train_app['CentralAir'][X_train_app['CentralAir'] == 'N'] = 0
    X_train_app['CentralAir'][X_train_app['CentralAir'] == 'Y'] = 1

    # Pushing all binary features to be of type int16
    X_train_app['CentralAir'] = X_train_app['CentralAir'].astype('int16')
    X_train_app['PoolArea'] = X_train_app['PoolArea'].astype('int16')
    X_train_app['GarageArea'] = X_train_app['GarageArea'].astype('int16')
    X_train_app['BsmtFinSF1'] = X_train_app['BsmtFinSF1'].astype('int16')
    X_train_app['Fireplaces'] = X_train_app['Fireplaces'].astype('int16')

    # One hot encode the neighbourhood feature
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
    X_ohe = ohe.fit_transform(X_train_app[['Neighborhood']])

    # Add the ohe columns and discard the object column
    X_train_app.drop(columns=['Neighborhood'], inplace=True)
    X_train_app = pd.concat([X_train_app, X_ohe], axis=1)

    return X_train_app


def preprocess_new_data(df):
    """
    This function preprocessed new data that comes from the API
    """
    # Create a dummy df to fit the neighbourhoods
    dummy_df = pd.DataFrame({'Neighborhood': NEIGHBOURHOODS})

    # One hot encode the neighbourhood feature
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
    ohe.fit(dummy_df[['Neighborhood']])
    X_ohe = ohe.transform(df[['Neighborhood']])

    # Add the ohe columns and discard the object column
    df.drop(columns=['Neighborhood'], inplace=True)
    df = pd.concat([df, X_ohe], axis=1)

    return df


def load_model():
    """
    This function loads the model
    """
    model_path = os.path.join(os.getcwd(), 'models', 'xgb_model2.pkl')

    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    return loaded_model
