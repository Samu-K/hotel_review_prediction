# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:11:07 2020

@author: Samu
"""

#Import in the needed funcs from functions.py
from functions import word_scorer, text_cleaner, unique_words, average_rating, unrated_value_drop, 

#We import these for data handling
import pandas as pd
import numpy as np

#Import sklearn for splitting data
from sklearn.model_selection import train_test_split

#Get data and split targets from it
fp = "D:/Coding/Rating analysis/data/tripadvisor_hotel_reviews.csv"
full_data = pd.read_csv(fp)

targets = full_data['Rating']
full_data.drop('Rating',axis=1,inplace=True)

#We make a copy to not alter the original df
train_c = full_data.copy()

#We set this new column to make it easier to append new values
train_c["unique_words"] = 0

#Loop through the index
for indx in train_c.index:
    #Clean the review and set the new value
    train_c.Review.loc[indx] = text_cleaner(train_c.Review.loc[indx])
    
    #Set a new column with a list of the unique words
    train_c["unique_words"][indx] = unique_words(train_c.Review.loc[indx])

#Split our data into training and validation
X_train,X_val,y_train,y_val = train_test_split(train_c,targets,test_size=0.33,random_state=0)

word_score = {}
for indx in X_train.index:
    #Assign ratings for each word
    words, rating = word_scorer(df=X_train,target_series=y_train,index=indx)
    for word in words:
        if word in word_score.keys():
            word_score[word].append(rating)
        else:
            word_score[word] = [rating]

#We first take the word_score dict and turn it into a pandas series
#Then we apply the average_rating function to get the avg rating for each word
#Then we drop Na values from it and finally sort it 
word_score_series = pd.Series(word_score_train).apply(average_rating).dropna().sort_values(ascending=False)

#Loop through all values and drop ones inbetween 2.4 and 4
for word, rating in word_score_series.items():
    if rating < 2.5 or rating >= 4:
        pass
    else:
        word_score_series = word_score_series.drop(index = word)

#Turn the unique_words into a list, instead of numpy array
X_train["unique_words"]=X_train.unique_words.apply(list)
X_val["unique_words"]=X_val.unique_words.apply(list)

#Drop any words in X_train and X_val that haven't been rated
#This avoids ValuErrors when predicting and gets rid of useless data
for index in X_train.index:
    data = X_train.loc[index, "unique_words"]
    new_words = unrated_value_drop(data,word_score_series)
    X_train.loc[index, "unique_words"] = new_words

for index in X_val.index:
    data = X_val.loc[index, "unique_words"]
    new_words = unrated_value_drop(data,word_score_series)
    X_val.loc[index, "unique_words"] = new_words

#Cut our data down, see Issues for more detail
cut_data_train = X_train[:2000]
cut_data_val = X_val[:2000]

#Encode the categorical variables into numerical with dummies.
#We first turn the unique_words into a series, 
#Then we stack and sum them into one line per review.
dummies_train = pd.get_dummies(cut_data_train.unique_words.apply(pd.Series).stack()).sum(level=0)
dummies_val = pd.get_dummies(cut_data_val.unique_words.apply(pd.Series).stack()).sum(level=0)

#We get rid of any words in the validation data that isn't in the training data
for column_name in dummies_val.columns:
    if column_name not in dummies_train.columns:
        dummies_val.drop(column_name,axis=1)

dummies_val = dummies_val.reindex(dummies_train.columns,axis=1)

X_train_ml = dummies_train
X_val_ml = dummies_val
y_train_ml = y_train[:2000]
y_val_ml = y_val[:2000]

X_train_ml.to_csv("D:/Coding/Rating analysis/data/X_train.csv")
X_val_ml.to_csv("D:/Coding/Rating analysis/data/X_val.csv")
y_train_ml.to_csv("D:/Coding/Rating analysis/data/y_train.csv")
y_val_ml.to_csv("D:/Coding/Rating analysis/data/y_val.csv")

X_train.to_csv("D:/Coding/Rating analysis/data/X_train_manual.csv")
X_val.to_csv("D:/Coding/Rating analysis/data/X_val_manual.csv")
y_train.to_csv("D:/Coding/Rating analysis/data/y_train_manual.csv")
y_val.to_csv("D:/Coding/Rating analysis/data/y_val_manual.csv")

word_score_series_train_copy.to_csv("D:/Coding/Rating analysis/data/word_score_series_train_copy.csv")

########
"""
This file creates the following files (of importance)
word_score_series_train_copy (contains ratings for words)
full_data (contains the original full data)
targets (contains the target values)
train_c (contains the modified data w/ unique_words)
X_train, X_val, y_train,y_val (split data)

"""
########
