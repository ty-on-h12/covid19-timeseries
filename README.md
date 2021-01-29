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

**I would like to predict new cases, but scraped data only has total (cummulative cases). By subtracting n-th+1 day from n-th I was able to create new cases column (had to drop the last row). Then I moved to visualization:** *note: all the sample charts are made from Poland's data.*<br>


 <img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/sample_chart.jpg' title="Sample charts">

**Cummulative cases make up an exponential curve, so I think that the best way to forecast this feature is to:**

* Transform data into logarythmic scale
* Fit a simple Linear Regression to log data
* Exponentialy transform predicted data

*In the later part of README I'll see how this works*

**New cases curve is much more unstable, I'll use ARIMA to forecast it.**

**Very important concept in time series prediction is *Stationarity***. Time series is stationary when its statistical properties don't change over time (mean, standard deviation stay roughly the same at all times). Most models assue that data is stationary, so before running statistical algorithms we must stationarize our data. Two components that help us define if a dataset is stationary or not are:

* **Trend** - General tendency over a long period of time, such as constantly increasing sales over 10 years.
* **Seasonality** - Is similar to trend, however instead on describing change over a long period of time it says if there are certain changes on a small subset of the general time period, such as increase in sales before christmas, this occurs each year, but isn't a general trend of increasing sales over 10 years.

**Python's Statmodels module provides neat functions that help to target this properties.**<br>

<img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/seasonal.jpg' title="Sample charts">


**Seasonality and trend are definitely visible. Other than *seeing* those properties we can look for their presence (or absence) using Dickey-Fuller test.** *details in jupyter notebook*

**Roling mean, rolling standard deviation plot**

<img src='https://github.com/ty-on-h12/covid19-timeseries/blob/master/utils/roll.jpg' title="Sample charts">

**From Dickey-Fuller test (notebook) and few plots, I can confidently say that data is not stationary. What now? How do we stationarize time series? There are few most relevant methods:**

* Taking a log or root
* Differentiation
* Exponential decay

**I am going to use log to stationarize the dataset.**

# Modeling

At first I was considering using a recurrent neural network for this forecast, but after reading <a href='https://arxiv.org/pdf/1909.00590.pdf'><b>this</b></a> great article and considering univariate nature of my dataset I've chosen to evaluate between **ARIMA** and **Prophet** (both models are covered in jupyter notebook). 

**You can see results on the website.**

# App deployment
**Only a few days ago I've found Streamlit (my previouse candidate was Dash), and after reading about it I've decided that this is the tool that I'll use for deployment.** For easy deployment of machine learning web apps Streamlit is very powerful, Dash may be more robust and generally better for big apps, but for my purpose with this project Streamlit was more than enough to satisfy my needs. Link to Streamlit's <a href='https://www.streamlit.io/'>website</a>. All the code for website is in app.py, global.py creates .csv for map chart, scrape.py scrapes data from Postman (which has a tendency to be unavaliable very frequently). If you would like to use this repo, just clone it, create folders: data and preds (scrape.py and global.py expect this folders to store data) and run my scripts. I belive I've commented my code well enough to get going right away :)

Link to the website : 
