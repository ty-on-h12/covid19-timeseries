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
    dataset = dataset.rename(columns={'Cases' : 'Cummulative_cases'})
    # Agreggating
    dataset = dataset.groupby(pd.Grouper(key='Date',freq='W-MON')).agg({'New_cases':'sum','Cummulative_cases':'sum'})
    dataset = dataset.reset_index()
    # Let the index be "Date"
    dataset = dataset.set_index('Date')
    # if for_model == 'ARIMA':
    # Stationarizing the dataset
    dataset = np.log(dataset)
    dataset = dataset.replace(np.NINF, np.NAN)
    # Interpolate to handle nans
    dataset['New_casse'] = dataset['New_cases'].interpolate(method='polynomial', order=3)
    dataset['Cummulative_cases'] = dataset['Cummulative_cases'].interpolate(method='linear')
    dataset = dataset.fillna(0)
    print(dataset.isna())
    return dataset['New_cases'], dataset['Cummulative_cases']

def create_model(data, future, order=(2,1,2)): 
    # Parametrized ARIMA, some datasets need bigger p to work properly
    model = ARIMA(data, order=order)
    model = model.fit()
    preds = model.predict(start=1, end=future, typ='levels')
    # Returning exp to reverse the log
    return np.exp(preds)


# REDUNDANT FUNCTION - REMOVE IN NEXT COMMIT 
def linear_regression(data, dates_to_predict):
    # Statsmodels implementation of linear regression
    # Converting dates to ordinal
    idx = pd.to_datetime(data.index)
    idx = [datetime.datetime.toordinal(date) for date in list(idx)]
    # Fitting the model
    lin_reg = sm.OLS(list(data.values), list(idx), formula='Cummulative_cases ~ np.power(Cummulative_cases, 2)')
    res = lin_reg.fit()
    # Converting dates to predict to ordinal
    dates_to_predict = pd.to_datetime(dates_to_predict)
    dates_to_predict = [datetime.datetime.toordinal(date) for date in list(dates_to_predict)]
    # Predicting
    preds = res.predict(np.squeeze(np.asarray(dates_to_predict)))
    # Back to date
    DATES = [datetime.datetime.fromordinal(date) for date in list(dates_to_predict)]
    # Results into series
    cummulative_series = pd.Series(data=np.exp(preds), index=DATES)
    return cummulative_series


def prophet(data, periods, ys=True):
    # Prophet requires to use special DataFrame template
    p_data = data.reset_index()
    cols = p_data.columns
    p_data = p_data.rename(columns={cols[0] : 'ds', cols[1] : 'y'})
    # Instantiating Prophet class
    prophet = Prophet(yearly_seasonality=ys, changepoint_prior_scale=0.1, interval_width=0.95)
    prophet.add_seasonality(name='monthly', period=30.5, fourier_order=3, prior_scale=0.1)
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

st.title('Welcome to Covid19 Forecast!')
st.write("""**Explore the future of Covid19 by playing around with interactive plots and statistical time series modeling methods: FBProphet and ARIMA.**""")
st.write("""**NOTE: This is the first release of this application, and there still may be some bugs, if you find any, please let me know via GitHub. All help appreciated :)**""")
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
    preprocessed, preprocessed_lin_reg = preprocess(DIR + countries_dict[country], for_model='ARIMA')
else:
    preprocessed, preprocessed_lin_reg = preprocess(DIR + countries_dict[country], for_model='Prophet')

st.write("""*Period* represents week since a country began to collect Covid19 statistics, for example period 50 means 50th week. 
Lowest bound is next week and the highest is next 12 weeks. 
**Higher bounds are not provieded since forecasts have much lower accuracy over a long period of time.** 
Data is spread on a **weekly** basis, meaning that each data point represents **sum** of either new or cummulative cases per week.""")
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
    dates_for_lin_reg = preds.index
    
    try:
        preds_cummulative = create_model(preprocessed_lin_reg, period)
    except ValueError:
        try:
            preds_cummulative = create_model(preprocessed_lin_reg, period, order=(4,1,2))
        except:
            preds_cummulative = create_model(preprocessed_lin_reg, period, order=(0,1,2))

    # Plotly figure new
    st.write("""Plot for **new** cases:""")
    fig = go.Figure()
    fig.add_scatter(x=preds.index, y=preds.values, name='Forecast')
    fig.add_scatter(x=preprocessed.index, y=np.exp(preprocessed.values), name='Real')
    fig.update_layout(template='simple_white', width=750, height=400, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_traces({"line":{'width':3}})
    st.plotly_chart(fig)

    # Plotly figure cummulative
    st.write("""Plot for **cummulative** cases:""")
    fig_reg = go.Figure()
    fig_reg.add_scatter(x=preds_cummulative.index, y=preds_cummulative, name='Forecast')
    fig_reg.add_scatter(x=preprocessed_lin_reg.index, y=np.exp(preprocessed_lin_reg.values), name='Real')
    fig_reg.update_layout(template='simple_white', width=750, height=400, margin=dict(l=20, r=20, t=20, b=20))
    fig_reg.update_traces({"line":{'width':3}})
    st.plotly_chart(fig_reg)

else:
    preds = prophet(preprocessed, period)
    fig = plot_plotly(preds[0], preds[1])
    fig.add_scatter(x=preprocessed.index, y=np.exp(preprocessed.values), name='Real')
    fig.update_layout(template='simple_white', width=750, height=400, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_traces({"line":{'width':1.5}})
    st.plotly_chart(fig)

    preds_cummulative = prophet(preprocessed_lin_reg, period, ys=False)    
    fig_2 = plot_plotly(preds_cummulative[0], preds_cummulative[1])
    fig_2.add_scatter(x=preprocessed_lin_reg.index, y=np.exp(preprocessed_lin_reg.values), name='Real')
    fig_2.update_layout(template='simple_white', width=750, height=400, margin=dict(l=20, r=20, t=20, b=20))
    fig_2.update_traces({"line":{'width':1.5}})
    st.plotly_chart(fig_2)
 

# Plotly map

st.title("Global forecast")
st.write("""**NOTE**: Due to computational efficiency global forecast is at the moment only available for ARIMA model.""")

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
    * $s(t)$ - Seasonality yearly, weekly or daily changes
    * $h(t)$ - Effects of holiday
    * $e(t)$ - Error term, used in case of unusual data   

    **For theory details I recommend visiting <a href='https://research.fb.com/blog/2017/02/prophet-forecasting-at-scale/'>Facebook's official site</a>.**
    **For code and implemenation details visit my <a href='https://github.com/ty-on-h12/covid19-timeseries'><b>GitHub repo</b></a>.**
    """, unsafe_allow_html=True)