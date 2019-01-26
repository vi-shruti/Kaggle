# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:59:31 2019

@author: Shruti
"""

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from datetime import datetime
import os
import csv

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column].tolist())
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

#Load the data as a dataframe and expand JSON columns using the defined function
train = load_df('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\train.csv')
test = load_df('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\test.csv')

#Saving raw expanded data
train.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\train_google_analytics_kaggle_full_expanded.csv', encoding='utf-8', index=False)
test.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\test_google_analytics_kaggle_full_expanded.csv', encoding='utf-8', index=False)

#Modifying transaction revenue to have no blanks and to be the "PredictedLogRevenue"
train['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
print(train['totals.transactionRevenue'].describe())

# Copy train into another dataframe to retain original data in train
all_data = train #.append(test).reset_index(drop=True, sort=True)
print(all_data.info())

#Finaf all columns with null values to complete the data
null_cnt = train.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])

# For the found columns, select and fillna object feature
for col in ['trafficSource.keyword',
            'trafficSource.referralPath',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.adContent']:
    all_data[col].fillna('unknown', inplace=True)

# fillna numeric feature
all_data['totals.pageviews'].fillna(1, inplace=True)
all_data['totals.newVisits'].fillna(0, inplace=True)
all_data['totals.bounces'].fillna(0, inplace=True)

#Define datatype explicitly
#all_data['fullVisitorId'] = all_data['fullVisitorId'].astype(int) #too long for int
#all_data['sessionId'] = all_data['sessionId'].astype(int) #too long for int
all_data['totals.pageviews'] = all_data['totals.pageviews'].astype(int)
all_data['totals.newVisits'] = all_data['totals.newVisits'].astype(int)
all_data['totals.bounces'] = all_data['totals.bounces'].astype(int)
all_data['visitId'] = all_data['visitId'].astype(int)
all_data['visitStartTime'] = all_data['visitStartTime'].astype(int)

#Set dates in proper format
all_data['date'] = all_data['date'].astype(datetime)
all_data['visitStartTime'] = all_data['visitStartTime'].astype(float)
#pd.Series(all_data['visitStartTime']).astype(float)
all_data['visitStartTime'] = pd.Timestamp(all_data['visitStartTime'], unit='s')
#all_data['visitStartTime'] = datetime.utcfromtimestamp(all_data['visitStartTime']).strftime('%Y-%m-%d %H:%M:%S')
all_data['visitStartTime'].head()

# fillna boolean feature
all_data['trafficSource.isTrueDirect'].fillna(False, inplace=True)

# Save modified data
all_data.to_csv('google_analytics_kaggle_full_expanded.csv', encoding='utf-8', index=False)

# Save modified data
train.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\train_google_analytics_kaggle_clean_expanded.csv', encoding='utf-8', index=False)
#test.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\test_google_analytics_kaggle_clean_expanded.csv', encoding='utf-8', index=False)

# Drop constant column
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)

all_data.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\google_analytics_kaggle_dropClean.csv', encoding='utf-8', index=False)
