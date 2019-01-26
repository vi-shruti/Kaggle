# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:35:42 2019

@author: Shruti
"""

import pandas as pd
#import numpy as np
#import json
#from pandas.io.json import json_normalize
#from datetime import datetime
#import os
#import csv

df = pd.read_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\google_analytics_kaggle_dropClean.csv',)

train_df = pd.get_dummies(df, columns=["channelGrouping", "device.browser", "device.deviceCategory", "device.isMobile", "device.operatingSystem", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.city", "geoNetwork.metro", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign", "trafficSource.isTrueDirect", "trafficSource.medium"], prefix=["channel", "devBrowser", "devCat", "devMob", "devOS", "geoCon", "geoCountry", "geoCity", "geoMetro", "geoReg", "geoSubCon", "adContent", "adNetworkType", "videoAd",  "page", "slot", "campaign", "trueDirect",  "medium"], sparse = True)
#trafficSource.keyword

columns = {"trafficSource.keyword", "trafficSource.referralPath", "trafficSource.source", "visitStartTime", "date", "geoNetwork.networkDomain", "trafficSource.adwordsClickInfo.gclId"}
train_df = train_df.drop(columns, axis=1)

train_df.to_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\google_analytics_kaggle_OneHotEncoded.csv', encoding='utf-8', index=False)