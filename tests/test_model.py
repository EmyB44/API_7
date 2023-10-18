import pytest
import pickle
import pandas as pd


def test_load_model():
    #Test loading model from pickle file

    with open("./predict_model_API.pkl", "rb") as file:
        model = pickle.load(file)


def test_predict():
    #Test predict fonction

    with open("./predict_model_API.pkl", "rb") as file:
        model = pickle.load(file)

    df= pd.read_csv('./X_train.csv')

    model.predict (df)

