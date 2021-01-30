import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import streamlit as st
from fbprophet import Prophet
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go 
import os
import datetime
import pycountry
from fbprophet.plot import plot_plotly, plot_components_plotly
warnings.filterwarnings('ignore')

# This script creates predicitons for each file in "data" folder 

DIR = './data/'
PREDS = './preds/'

def preprocess(data, for_model):
    # transforming for modeling
    dataset = pd.read_csv(data)
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    # Loop to create new cases from cummulative ones
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
    dataset = dataset.rename(columns={'Cases' : 'Cumulative_cases'})
    # Agreggating
    dataset = dataset.groupby(pd.Grouper(key='Date',freq='W-MON')).agg({'New_cases':'sum','Cumulative_cases':'sum'})
    dataset = dataset.reset_index()
    # Let the index be "Date"
    dataset = dataset.set_index('Date')
    # Stationarizing the dataset
    # if for_model=='ARIMA':
    dataset = np.log(dataset)
    dataset = dataset.replace(np.NINF, np.NAN)
    # Interpolate to handle nans
    dataset['New_casse'] = dataset['New_cases'].interpolate(method='polynomial', order=3)
    dataset['Cumulative_cases'] = dataset['Cumulative_cases'].interpolate(method='linear')
    dataset = dataset.fillna(0)
    print(dataset.isna())
    return dataset['New_cases'], dataset['Cumulative_cases']

def create_model(data, future, order=(2,1,2)): 
    # Parametrized ARIMA, some datasets need bigger p to work properly
    model = ARIMA(data, order=order)
    model = model.fit()
    preds = model.predict(start=1, end=future, typ='levels')
    # Returning exp to reverse the log
    return np.exp(preds)

def prophet(data, periods, ys=True):
    # Prophet requires to use special DataFrame template
    p_data = data.reset_index()
    cols = p_data.columns
    p_data = p_data.rename(columns={cols[0] : 'ds', cols[1] : 'y'})
    # Instantiating Prophet class
    prophet = Prophet(yearly_seasonality=ys, changepoint_prior_scale=0.1, interval_width=0.95)
    prophet.add_seasonality(name='monthly', period=30, fourier_order=3, prior_scale=0.1)
    prophet.fit(p_data)
    future = prophet.make_future_dataframe(periods=periods)
    forecast = prophet.predict(future)
    temp = forecast['ds']
    forecast = forecast[['yhat', 'yhat_lower', 'yhat_upper']].apply(np.exp)
    forecast['ds'] = temp
    return prophet, forecast

# STREAMLIT

countries = tuple(list(os.walk(DIR))[0][2])
countries_map = [x.replace('-', ' ')[:-4].capitalize() for x in countries]
countries_dict = dict(zip(countries_map, countries))

st.title('Covid19 Forecast')
st.write("""**Explore the future of Covid19 by playing around with interactive plots made using time series modeling methods: FBProphet and ARIMA.**""")
st.write("""**NOTE: I do not claim my forecasts to be very accurate, in fact some of them are completly off.**""")
st.markdown("For code <a href='https://github.com/ty-on-h12/covid19-timeseries'><b>check out my GitHub</b></a>", unsafe_allow_html=True)
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
MODEL  = st.selectbox('Select model', ('ARIMA', 'Prophet'))

if MODEL == 'ARIMA':
    preprocessed, preprocessed_c = preprocess(DIR + countries_dict[country], for_model='ARIMA')
else:
    preprocessed, preprocessed_c = preprocess(DIR + countries_dict[country], for_model='Prophet')

st.write("""*Period* represents week since a country began to collect Covid19 statistics, for example period 50 means 50th week. 
**Minimal forecast time is 1 week , maximal 12th week.""")
period = st.slider('Select period', min_value=len(preprocessed), max_value=len(preprocessed)+12)

st.title('Forcast for {}'.format(country))

if MODEL == 'ARIMA':
    try:
        preds = create_model(preprocessed, period)
    except ValueError:
        try:
            preds = create_model(preprocessed, period, order=(4,1,2))
        except:
            preds = create_model(preprocessed, period, order=(0,1,2))
    DATES = preds.index
    
    try:
        preds_cumulative = create_model(preprocessed_c, period)
    except ValueError:
        try:
            preds_cumulative = create_model(preprocessed_c, period, order=(4,1,2))
        except:
            preds_cumulative = create_model(preprocessed_c, period, order=(0,1,2))

    # Plotly figure new
    st.write("""Plot for **new** cases:""")
    fig = go.Figure()
    fig.add_scatter(x=preds.index, y=preds.values, name='Forecast')
    fig.add_scatter(x=preprocessed.index, 
        y=np.exp(preprocessed.values), 
        name='Real') 
        # labels={preprocessed.index:'Date', preprocessed['']:"New cases"})
    fig.update_layout(template='simple_white', width=750, height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_traces({"line":{'width':3}})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Plotly figure cumulative
    st.write("""Plot for **cumulative** cases:""")
    fig_2 = go.Figure()
    fig_2.add_scatter(x=preds_cumulative.index, y=preds_cumulative, name='Forecast')
    fig_2.add_scatter(x=preprocessed_c.index, 
        y=np.exp(preprocessed_c.values), 
        name='Real')
        # labels={preprocessed_c.index:'Date', preprocessed_c.values:"New cases"})
    fig_2.update_layout(template='simple_white', width=750, height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    # fig_2.update_xaxes(rangeslider_visible=True)
    fig_2.update_traces({"line":{'width':3}})
    st.plotly_chart(fig_2, use_container_width=True, config={"displayModeBar": False})

else:
    # Plotly figure new
    preds = prophet(preprocessed, period)
    preds[0].history['y'] = np.exp(preds[0].history['y']) 
    fig = plot_plotly(preds[0], preds[1], xlabel='', ylabel='')
    fig.update_layout(template='simple_white', width=750, height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_traces({"line":{'width':1.5}})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Plotly figure cummulative
    preds_cumulative = prophet(preprocessed_c, period, ys=False)
    preds_cumulative[0].history['y'] = np.exp(preds_cumulative[0].history['y'])     
    fig_2 = plot_plotly(preds_cumulative[0], preds_cumulative[1], xlabel='', ylabel='')
    fig_2.update_layout(template='simple_white', width=750, height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    fig_2.update_xaxes(rangeslider_visible=False)
    fig_2.update_traces({"line":{'width':1.5}})
    st.plotly_chart(fig_2, use_container_width=True, config={"displayModeBar": False})

# Plotly map

st.title("Global forecast")
st.write("**Zoom, rotate and mouseover to interact with the map!**")
# st.write("""**NOTE**: Due to computational efficiency global forecast is at the moment only available for ARIMA model.""")

df = pd.read_csv(PREDS + 'global.csv')
full_countries = [cmap for cmap, c in countries_dict.items()]

iso_map = {}
for country in pycountry.countries:
    iso_map[country.name] = country.alpha_3
codes = [iso_map.get(country, np.NaN) for country in full_countries]
df['iso_alpha'] = df['country'].map(dict(zip(full_countries, codes)))
df = df[pd.to_datetime(df['date']) > '2020-05-01']
df = df.dropna()

fig_3 = px.choropleth(df, locations="iso_alpha",
                     hover_name="country", 
                     color="cases",
                     animation_frame="date",
                     color_continuous_scale='Blues',
                     projection="natural earth", height=400)
fig_3.update_layout(margin=dict(l=20, r=20, t=0, b=0), coloraxis_showscale=False)
st.plotly_chart(fig_3, use_container_width=True, config={"displayModeBar": False})

# Markdown

if MODEL == 'ARIMA':
    st.title('Modeling data with ARIMA')
    st.markdown("""
    **ARIMA** is a statistical method of forecasting time series data. 
    It requires it's input data to be **stationary** - data which lacks **trend** and **seasonality**.

    * Trend - General tendency present over a **long period** of time in data, such as increasing sales over a period of 10 years. Mathematically speaking trend is change in data's **rolling mean**.
    * Seasonality - **Occasional** tendency to visibly increase or decrease, example - sales increasing each year before christmas. **Rolling standard deviation** differs over time.
    In other words one can say time series is stationary when it's **statistical properties** don't change over time.
    """)
    st.title('ARIMA - components')
    st.markdown(r"""
    **ARIMA** stands for:

    * AR - Auto Regressive (assumes that previous values affect current ones). Formula: $y_{t}=B_{1}y_{t-1} + B_{2}y_{t-2} + ... + B_{n}y_{t-n}$ - essentialy Linear Regression with lagged $y$
    * I - Integrated (helps with stationarizng time series). Formula (for trend elimination): $I_{t}=y_{t+1} - y_{t}$
    * MA - Moving average (assumes that current value of $y$ is influenced by previous days term error). Formula: $y_{t}=\mu \epsilon + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + ... + \theta_{n}\epsilon_{t-n}$

    **ARIMA** combines ideas from all three of those concepts. Details can be found in my <a href='https://github.com/ty-on-h12/covid19-timeseries'><b>GitHub repo</b></a>.
    """, unsafe_allow_html=True)
else:
    st.title('Modeling with Prophet')
    st.markdown("""
    **Prophet** is forecasting model developed by Facebook's Data Science team. 
    It's an open source projects and since it's release it became one of most widely used forecasting algorithms.
    Prophet's biggest strengths are it's speed (it's written in Stan) and ease of use. Prophet requires very little **data preprocessing**.
    """)
    st.title('Prophet - components')
    st.markdown("""
    Prophet uses several functions to produce results - $y(t)=g(t)+s(t)+h(t)+e(t)$ where:
    
    * $g(t)$ - Linear or logistic curve used for data modeling
    * $s(t)$ - Seasonality - yearly, weekly or daily changes
    * $h(t)$ - Effects of holiday
    * $e(t)$ - Error term, used in case of unusual data   

    **For theory details I recommend visiting <a href='https://research.fb.com/blog/2017/02/prophet-forecasting-at-scale/'>Facebook's official site</a>.**

    **For code and implemenation details visit my <a href='https://github.com/ty-on-h12/covid19-timeseries'><b>GitHub repo</b></a>.**
    """, unsafe_allow_html=True)

st.write("""**If've found any bugs, feel free to let me know via GitHub.**""")