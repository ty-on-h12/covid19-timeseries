from app import create_model, preprocess, countries
import pandas as pd
import numpy as np
import os

# This script is used to create global predictions for map chart in the main app

# Necessary dirs
DIR = './data/'
PREDS = './preds/'
# Countries list
countries = tuple(list(os.walk(DIR))[0][2])
countries_map = [x.replace('-', ' ')[:-4].capitalize() for x in countries]
countries_dict = dict(zip(countries_map, countries))
# Base DataFrame
df = pd.DataFrame(columns=['cases', 'country', 'date'])
# List for unavailable countries
unavailable = []
# Loop to compute forecasts
# Lots of missing data thus try/excepts
for c in countries:
    try:
        pre = preprocess(DIR + c)
    except ValueError:
        unavailable.append(c)
        pass
    try:
        preds = create_model(pre, len(pre)+12)
        temp_df = pd.DataFrame(columns=['cases', 'country'])
        temp_df['cases'] = preds.values
        temp_df['country'] = c[:-4].capitalize()
        temp_df['date'] = preds.index
    except:
        try:
            preds = create_model(pre, len(pre)+12, 4)
            temp_df = pd.DataFrame(columns=['cases', 'country'])
            temp_df['cases'] = preds.values
            temp_df['country'] = c[:-4].capitalize()
            temp_df['date'] = preds.index
        except:
            unavailable.append(c)
            pass
    df = df.append(temp_df, ignore_index=True)
# Data nicely organized and put into csv
df = df.set_index('date')
df.to_csv(PREDS + 'global.csv')
print(unavailable)