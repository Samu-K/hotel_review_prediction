# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:46:47 2020

@author: Samu
"""
import pandas as pd
import ast
import numpy as np

needed_data = ["X_train","X_val",
               "y_train","y_val",
               "word_score_series_train_copy"]

filepaths = []
for data in needed_data:
    fp = "D:/Coding/Rating analysis/data/" + data + ".csv"
    filepaths.append(fp)
    
X_train = pd.read_csv(filepaths[0],index_col=0,converters={2:ast.literal_eval})
X_val = pd.read_csv(filepaths[1],index_col=0,converters={2:ast.literal_eval})
y_train = pd.read_csv(filepaths[2],index_col=0)
y_val = pd.read_csv(filepaths[3],index_col=0)
word_score_series_train_copy = pd.read_csv(filepaths[4],index_col=0,dtype={1:str,2:np.float64},squeeze=True)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

parameters = {
'n_estimators': 350,
'max_depth': 6,
'min_child': 1,
'l_rate': 0.15,
'gamma': 0}

for i in range(0,10):
    parameters['gamma'] = i
    optimized_model = XGBRegressor(random_state=0,
                                   n_estimators=parameters['n_estimators'],
                                   max_depth=parameters['max_depth'],
                                   min_child_weight=parameters['min_child'],
                                   learning_rate=parameters['l_rate'],
                                   gamma = parameters['gamma'],
                                   nthread=16)
        
    optimized_model.fit(X_train,y_train)
    preds = optimized_model.predict(X_val)
    mae = mean_absolute_error(y_val,preds)
    print(i, mae)