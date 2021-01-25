import requests
import json
import pandas as pd
import numpy as np
import time

UNAVAILABLE = []

# Export funtion => exports DataFrames to CSV files for further use in data analysis
def export(dataframe):
    # Each CSV file is going to be named after it's country
    name = dataframe['Country'].unique()[0].lower()
    if ' ' in name:
        name = dataframe['Country'].unique()[0].replace(' ', '_').lower() 
    dataframe.to_csv('./data/{}.csv'.format(name))

# Funtion to scrape data and put it nicely in a Pandas DataFrame
def scrape_data(country):
    # Sleep, becouse there is a limit on requests 
    time.sleep(3)
    # Try/Except becouse requesting sometimes throws an error ( due to limits )
    try:
        # Step I - get the API's url, format for different countries 
        URL = 'https://api.covid19api.com/total/dayone/country/{}/status/confirmed'
        # Step II - get request to the url, response to JSON
        res = requests.get(URL.format(country)).json()
        # Step III - List comprehension to get lists of 1. Cases, 2. Dates, 3. Current country
        CASES = [res[i]['Date'][:10] for i in range(len(res))]
        DATE = [res[i]['Cases'] for i in range(len(res))]
        COUNTRY = [res[i]['Country'] for i in range(len(res))]
        # Step IV - Creating the DataFrame with "Date" and "Cases" columns 
        df = pd.DataFrame(list(zip(DATE, CASES)), columns=['Date', 'Cases'])
        # Adding "Country" column
        df['Country'] = COUNTRY
        print('\n' + 10*'=' + f' {country} dataframe created ' + 10*'=')
        export(df)
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

main()


