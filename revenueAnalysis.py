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

# Function importing Dataset 
def importdata(): 
    df = pd.read_csv('C:\\Users\\Shruti\\Documents\\GoogleAnalyticsCustomerRevenuePrediction\\google_analytics_kaggle_dropClean.csv')
    df['fullVisitorId'] = df['fullVisitorId'].astype('float32')
    df['sessionId'] = df['sessionId'].astype('float32')
    df['visitId'] = df['visitId'].convert_objects(convert_numeric=True)
    #dtype={'fullVisitorId': np.float64, 'sessionId': np.float64, 'visitId': np.float64},)

    balance_data = pd.get_dummies(df, columns=["channelGrouping", "device.browser", "device.deviceCategory", "device.isMobile", "device.operatingSystem", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.city", "geoNetwork.metro", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign", "trafficSource.isTrueDirect", "trafficSource.medium"], prefix=["channel", "devBrowser", "devCat", "devMob", "devOS", "geoCon", "geoCountry", "geoCity", "geoMetro", "geoReg", "geoSubCon", "adContent", "adNetworkType", "videoAd",  "page", "slot", "campaign", "trueDirect",  "medium"], sparse = True)

    columns = {"trafficSource.keyword", "trafficSource.referralPath", "trafficSource.source", "visitStartTime", "date", "geoNetwork.networkDomain", "trafficSource.adwordsClickInfo.gclId"}
    balance_data = balance_data.drop(columns, axis=1)
  
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
  
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 0:33] 
    Y = balance_data.values[:, 34]
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=4, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 4, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
  
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    dot_data = tree.export_graphviz(clf_gini,
                                #feature_names=X_test.feature_names,
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
      
# Calling main function 
if __name__=="__main__": 
    main() 