# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:48:21 2020

@author: Samu
"""
#First let's import in the data we need
from functions import *
import ast

needed_data = ["X_train_manual","X_val_manual","y_train_manual","y_val_manual","word_score_series_train_copy"]
filepaths = []

for data in needed_data:
    fp = "D:/Coding/Rating analysis/data/" + data + ".csv"
    filepaths.append(fp)
    
X_train = pd.read_csv(filepaths[0],index_col=0,converters={2:ast.literal_eval})
X_val = pd.read_csv(filepaths[1],index_col=0,converters={2:ast.literal_eval})
y_train = pd.read_csv(filepaths[2],index_col=0)
y_val = pd.read_csv(filepaths[3],index_col=0)
word_score_series_train_copy = pd.read_csv(filepaths[4],index_col=0,dtype={1:str,2:np.float64},squeeze=True)

preds = {}

#We go through the index, so that we can get the matching parts of each df
for index in X_val.index:
    word_values = word_score_series_train_copy
    words = X_val.unique_words[index]
    text = text_cleaner(X_val.Review[index])
    word_count = unique_word_count(text,words)
    pred = review_predictor(word_values,words,word_count)
    #Save the prediction into the preds dict,by index
    preds[index] = pred


error = {}
#Go through all the predictions
for index,pred in preds.items():
    if pred == None:
        continue
    target = y_val.Rating.loc[index]
    #Get the difference between the real review and the prediction
    abs_error = abs(target-pred)
    #Round it to 4 decimals and save into preds by index
    error[index] = round(abs_error,4)

#This gets us the average error from all the predictions
avg_error = sum(error.values()) / len(error)
print("The average error is:" + str(avg_error))

text = "Comfy Bed, very clean large room ideally situated for a short break or a longer stay.  Joanne and John manage the property extremely well and they have a fantastic group of friendly staff.  We have stayed here a number of times and it feels like coming home when we visit.  Excellent value for money.  The restaurant has a limited menu but the food is excellent reasonably priced with large portions.  Breakfast is great value."
pred = predict(text,word_values=word_score_series_train_copy)
print("Prediction for the review is :" + str(pred))
