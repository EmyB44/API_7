import pytest
import pandas as pd




def test_load_data():
# test loading data from csv file
    df= pd.read_csv('./X_train.csv')
    df.iloc[0]



