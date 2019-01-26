# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:07:29 2019

@author: Shruti
"""

#Google Analytics Revenue Analysis
# Importing the required packages 
#import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import pydotplus
from sklearn import tree
import collections

import os
os.environ['PATH'].split(os.pathsep)
os.environ['PATH'] += os.pathsep + 'C:\\Users\\Shruti\\Anaconda3\\Library\\bin\\graphviz'

     
# Building Phase 
df = pd.read_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\google_analytics_kaggle_dropClean.csv')
df['fullVisitorId'] = df['fullVisitorId'].astype('float32')
df['sessionId'] = df['sessionId'].astype('float32')
df['visitId'] = df['visitId'].astype('float32')

balance_data = pd.get_dummies(df, columns=["channelGrouping", "device.browser", "device.deviceCategory", "device.isMobile", "device.operatingSystem", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.city", "geoNetwork.metro", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign", "trafficSource.isTrueDirect", "trafficSource.medium"], prefix=["channel", "devBrowser", "devCat", "devMob", "devOS", "geoCon", "geoCountry", "geoCity", "geoMetro", "geoReg", "geoSubCon", "adContent", "adNetworkType", "videoAd",  "page", "slot", "campaign", "trueDirect",  "medium"], sparse = True)

columns = {"trafficSource.keyword", "trafficSource.referralPath", "trafficSource.source", "visitStartTime", "date", "geoNetwork.networkDomain", "trafficSource.adwordsClickInfo.gclId"}
balance_data = balance_data.drop(columns, axis=1)
feature_names = balance_data.columns.values.tolist()

#Split Data
X = balance_data.values[:, 0:33] 
Y = balance_data.values[:, 34]
  
# Spliting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(  
X, Y, test_size = 0.3, random_state = 100) 

clf_gini = DecisionTreeClassifier(criterion = "gini", 
        random_state = 100,max_depth=4, min_samples_leaf=5) 
  
# Performing training 
clf_gini.fit(X_train, y_train) 

clf_entropy = DecisionTreeClassifier( 
        criterion = "entropy", random_state = 100, 
        max_depth = 4, min_samples_leaf = 5) 
  
# Performing training 
clf_entropy.fit(X_train, y_train) 

# Operational Phase 
print("Results Using Gini Index:") 
  
# Prediction using gini 
y_pred_gini = clf_gini.predict(X_test) 
print("Predicted values:") 
print(y_pred_gini) 
print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred_gini)) 
  
print ("Accuracy : ", 
accuracy_score(y_test,y_pred_gini)*100) 
  
print("Report : ", 
classification_report(y_test, y_pred_gini)) 
  
print("Results Using Entropy:") 
# Prediction using entropy 
y_pred_entropy = clf_entropy.predict(X_test) 
print("Predicted values:") 
print(y_pred_entropy) 

print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred_entropy)) 
  
print ("Accuracy : ", 
accuracy_score(y_test,y_pred_entropy)*100) 
  
print("Report : ", 
classification_report(y_test, y_pred_entropy)) 

dot_data = tree.export_graphviz(clf_gini,
                                feature_names=feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')      
