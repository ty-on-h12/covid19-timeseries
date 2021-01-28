import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import streamlit as st
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go 
import os
import datetime
import pycountry
warnings.filterwarnings('ignore')

# This script creates predicitons for each file in "data" folder 

DIR = './data/'
PREDS = './preds/'

def preprocess(data):
    # transforming for modeling
    dataset = pd.read_csv(data)
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    # Interpolate to handle nans
    new_cases = []
    for n in range(dataset.shape[0]):
        try:
            new_cases.append(dataset['Cases'][n+1] - dataset['Cases'][n])
        except KeyError:
            # new_cases.append(np.NaN) - would contain nan, so let's just pass
            pass
    # Becouse new_cases is shorter by 1 we need to drop last row of the dataset
    dataset.drop(dataset.tail(1).index, inplace=True)
    dataset['New_cases'] = new_cases
    dataset = dataset.rename(columns={'Cases' : 'Cummulative_cases'})
    dataset.drop(['Cummulative_cases'], axis=1, inplace=True)
    # Agreggating
    dataset = dataset.groupby(pd.Grouper(key='Date',freq='W-MON')).agg({'New_cases':'sum'})
    dataset = dataset.reset_index()
    # Let the index be "Date"
    dataset = dataset.set_index('Date')
    # Stationarizing the dataset
    dataset = np.log(dataset)
    dataset = dataset.replace(np.NINF, np.NAN)
    dataset = dataset['New_cases'].interpolate(method='polynomial', order=3)
    dataset = dataset.fillna(0)
    return dataset

def create_model(data, future, ar=2): 
    # Parametrized ARIMA, some datasets need bigger p to work properly
    model = ARIMA(data, order=(ar,1,2))
    model = model.fit()
    preds = model.predict(start=1, end=future, typ='levels')
    # Returning exp to reverse the log
    return np.exp(preds)

def csv(preds, directory):
    preds.to_csv(directory, index=False)

# STREAMLIT

countries = tuple(list(os.walk(DIR))[0][2])
countries_map = [x.replace('-', ' ')[:-4].capitalize() for x in countries]
countries_dict = dict(zip(countries_map, countries))

st.title('Covid19 Forecast')
st.markdown("<a href='https://github.com/ty-on-h12/covid19-timeseries'><b>Check out my GitHub</b></a>", unsafe_allow_html=True)
st.write('Forecast available for {} countries!'.format(len(countries_map)))
usr_input = st.text_input('Check if a country is available', value='')

if len(usr_input) > 1:
    if usr_input.capitalize() in countries_map:
        st.write(usr_input.capitalize() + ' is avaliable')
    else:
        st.write(usr_input.capitalize() + " isn't avaliable")
else:
    pass

country = st.selectbox('Select country', (sorted(countries_map)))

preprocessed = preprocess(DIR + countries_dict[country])
st.write("""*Period* represents week since a country began to collect Covid19 statistics, for example period 50 means 50th week. 
Lowest bound is next week and the highest is next 12 weeks. 
**Higher bounds are not provieded since forecasts have much lower accuracy over a long period of time**""")
period = st.slider('Select period', min_value=len(preprocessed), max_value=len(preprocessed)+12)
try:
    preds = create_model(preprocessed, period)
except ValueError:
    preds = create_model(preprocessed, period, ar=4)
st.write('Forcast for {}'.format(country))

# Plotly figure
fig = go.Figure()
fig.add_scatter(x=preds.index, y=preds.values, name='Forecast')
fig.add_scatter(x=preprocessed.index, y=np.exp(preprocessed.values), name='Real')
fig.update_layout(template='simple_white', width=750, height=400, margin=dict(l=20, r=20, t=20, b=20))
fig.update_traces({"line":{'width':3}})
st.plotly_chart(fig)


# Plotly map

st.write("Global forecast")

df = pd.read_csv(PREDS + 'global.csv')
full_countries = [cmap for cmap, c in countries_dict.items()]

iso_map = {}
for country in pycountry.countries:
    iso_map[country.name] = country.alpha_3
codes = [iso_map.get(country, np.NaN) for country in full_countries]
df['iso_alpha'] = df['country'].map(dict(zip(full_countries, codes)))
df = df[pd.to_datetime(df['date']) > '2020-05-01']
df = df.dropna()

fig_2 = px.choropleth(df, locations="iso_alpha",
                     hover_name="country", 
                     color="cases",
                     animation_frame="date",
                     color_continuous_scale='Blues',
                     projection="natural earth")
fig_2.update_layout(margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_2)