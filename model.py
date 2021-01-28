import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

# This script creates predicitons for each file in "data" folder 

DIR = './data/'
PREDS = './preds/'

def preprocess(data):
    # transforming for modeling
    dataset = pd.read_csv(data)
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    # 
    new_cases = []
    for n in range(dataset.shape[0]):
        try:
            new_cases.append(dataset['Cases'][n+1] - dataset['Cases'][n])
        except KeyError:
            # new_cases.append(np.NaN) - would contain nan, so let's just pass
            pass
    #
    dataset.drop(dataset.tail(1).index, inplace=True)
    dataset['New_cases'] = new_cases
    dataset = dataset.rename(columns={'Cases' : 'Cummulative_cases'})
    dataset.drop(['Cummulative_cases'], axis=1, inplace=True)
    # 
    dataset = dataset.groupby(pd.Grouper(key='Date',freq='W-MON')).agg({'New_cases':'sum'})
    dataset = dataset.reset_index()
    # Let the index be "Date"
    dataset = dataset.set_index('Date')
    # Stationarizing the dataset
    dataset = np.log(dataset)
    return dataset

def create_model(data, future): 
    model = ARIMA(data, order=(2,1,2))
    model = model.fit()
    preds = model.predict(start=1, end=future, typ='levels')
    return np.exp(preds)

def csv(preds, directory):
    preds.to_csv(directory, index=False)

country = preprocess(DIR + 'poland.csv')
country_preds = create_model(country, 60)
csv(country_preds, PREDS + 'test.csv')