import requests
import json
import pandas as pd
import numpy as np
import time
from requests.exceptions import ConnectionError

# List of unavailable countries
UNAVAILABLE = []

# Export funtion => exports DataFrames to CSV files for further use in data analysis
def export(dataframe, country):
    # Each CSV file is going to be named after it's country
    dataframe.drop(['Unnamed: 0'], axis=1 ,inplace=True)
    dataframe.to_csv('./data/{}.csv'.format(country))

# Funtion to scrape data and put it nicely in a Pandas DataFrame
def scrape_data(country):
    # Sleep, becouse there is a limit on requests 
    time.sleep(3)
    # Step I - get the API's url, format for different countries 
    URL = 'https://api.covid19api.com/total/dayone/country/{}/status/confirmed'
    # Some URLs are unavailable, so Try/Except
    try:
        # Step II - get request to the url, response to JSON
        res = requests.get(URL.format(country)).json()
    except ConnectionError or ValueError as err:
        print(err)
        print(URL + ' is unavailable')
        UNAVAILABLE.append(country)
    # Try/Except becouse requesting sometimes throws an error ( due to limits )
    try:
        # Step III - List comprehension to get lists of 1. Cases, 2. Dates, 3. Current country
        CASES = [res[i]['Date'][:10] for i in range(len(res))]
        DATE = [res[i]['Cases'] for i in range(len(res))]
        # Step IV - Creating the DataFrame with "Date" and "Cases" columns 
        df = pd.DataFrame(list(zip(CASES, DATE)), columns=['Date', 'Cases'])
        print('\n' + 10*'=' + f' {country} dataframe created ' + 10*'=')
        export(df, country)
        print('\n' + 10*'=' + f' {country} dataframe exported ' + 10*'=')
    except KeyError or IndexError as err:
        print(err)
        print(country)
        UNAVAILABLE.append(country)

# The main function => creates dataframe for all the countries
def main():
    # First i want to get the list of all available countries 
    # Next 3 lines create that list
    URL = 'https://api.covid19api.com/countries'
    res = requests.get(URL).json()
    global COUNTRIES
    COUNTRIES = [res[i]['Slug'] for i in range(len(res))]
    # Creating the DataFrames
    for country in COUNTRIES:
        scrape_data(country)
    print(UNAVAILABLE)

main()