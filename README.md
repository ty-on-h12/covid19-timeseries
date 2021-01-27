# Covid19 - Time series forecasting

**Table of contents:**
* Projet overview
* Data exploration
* Modeling
* App deployment

# Project overview
**Covid19 forecasting build using ARIMA / Prophet. Using data scraped from Postman. Deployed on Streamlit.**

**Introduction**

Some time ago I've found <a href='https://documenter.getpostman.com/view/10808728/SzS8rjbc'><b>this</b></a> API, 
it's structure is perfect for some JavaScript web app, but with <a href='https://requests.readthedocs.io/en/master/'><i>requests</i></a> and <a href='https://pandas.pydata.org/'><i>pandas</i></a> I scraped it and I've put it into DataFrames, so this data can be easily manipulated and modeled. Data is organized into csv files - each file for each country. I've done it this way, becouse with files organized in such structure it is easy to create machine learning pipelines. <br> **Pipeline overview:**
* Data preprocessing and aggregation
* Model selection and evaluation 
* Training
* Deploying<br>

*Check jupyter notebook for details.*

*If you would like you can get that data by creating folder **data** and running **scrape.py** - scraping takes some time due to request limit from Postman.*

# Data exploration
CASES CHARTS: 
<p align='center'>
  <img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/sample_chart.jpg' title="Sample charts">
</p>
<p align='center'>
  <img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/seasonal.jpg' title="Sample charts">
</p>
<p align='center'>
  <img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/roll.jpg' title="Sample charts">
</p>
AUTOCORRELATION CHARTS:

# Modeling

At first I was considering using a recurrent neural network for this forecast, but after reading <a href='https://arxiv.org/pdf/1909.00590.pdf'><b>this</b></a> great article and considering univariate nature of my dataset I've chosen to evaluate between **ARIMA** and **Prophet** ( both models are covered in jupyter notebook ).

# App deployment
