# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:46:47 2020

@author: Samu
"""
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


needed_data = ["X_train","X_val",
               "y_train","y_val"]

filepaths = []
for data in needed_data:
    fp = "data/" + data + ".csv"
    filepaths.append(fp)
    
X_train = pd.read_csv(filepaths[0],index_col=0,converters={2:ast.literal_eval})
X_val = pd.read_csv(filepaths[1],index_col=0,converters={2:ast.literal_eval})
y_train = pd.read_csv(filepaths[2],index_col=0)
y_val = pd.read_csv(filepaths[3],index_col=0)

combined_data = pd.concat([X_train,X_val],sort=True)
combined_targets = pd.concat([y_train,y_val],sort=True)

combined_data.unique_words = combined_data.unique_words.apply(lambda x:" ".join(x))

full_vect = CountVectorizer()
full_vect = full_vect.fit_transform(combined_data.unique_words)

X_train,X_val,y_train,y_val = train_test_split(full_vect,combined_targets,random_state=0)

parameters = {
'n_estimators': 450,
'max_depth': 6,
'min_child': 1,
'l_rate': 0.2}

model = XGBRegressor(random_state=0,
                                n_estimators=parameters['n_estimators'],
                                max_depth=parameters['max_depth'],
                                min_child_weight=parameters['min_child'],
                                learning_rate=parameters['l_rate'],
                                nthread=16)
        
model.fit(X_train,y_train)
preds = model.predict(X_val)
mae = mean_absolute_error(y_val,preds)
print(mae)
