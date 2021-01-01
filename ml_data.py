# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:11:07 2020

@author: Samu
"""

#Import in the needed funcs from functions.py
from functions import word_scorer, text_cleaner, unique_words, average_rating, unrated_value_drop 

#We import these for data handling
import pandas as pd

#Import sklearn for splitting data
from sklearn.model_selection import train_test_split

#To create the datafolder, if it doesn't exist
import os

#Get data and split targets from it
fp = "tripadvisor_hotel_reviews.csv"
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
    train_c.loc[indx,"Review"] = text_cleaner(train_c.loc[indx,"Review"])
    
    #Set a new column with a list of the unique words
    train_c["unique_words"].loc[indx] = unique_words(train_c.loc[indx,"Review"])

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
word_score_series = pd.Series(word_score).apply(average_rating).dropna().sort_values(ascending=False)

#Loop through all values and drop ones inbetween 2.4 and 4
for word, rating in word_score_series.items():
    if rating < 2.5 or rating >= 4:
        pass
    else:
        word_score_series = word_score_series.drop(index = word)

#Turn the unique_words into a list, instead of numpy array
X_train.unique_words=X_train.unique_words.apply(list)
X_val.unique_words=X_val.unique_words.apply(list)

#Drop any words in X_train and X_val that haven't been rated
#This avoids ValuErrors when predicting and gets rid of useless data
for index in X_train.index:
    data = X_train.at[index, "unique_words"]
    new_words = unrated_value_drop(data,word_score_series)
    X_train.at[index, "unique_words"] = new_words

for index in X_val.index:
    data = X_val.at[index, "unique_words"]
    new_words = unrated_value_drop(data,word_score_series)
    X_val.at[index, "unique_words"] = new_words

#Check if the datafolder exists
if os.path.isdir("data") == False:
    #If not create the datafolder
    os.makedirs("data")
else:
    #If it does exist, we clear it out and create a new one
    os.removedirs("data")
    os.makedirs("data")


#Export our data out
X_train.to_csv("data/X_train.csv",header=True)
X_val.to_csv("data/X_val.csv",header=True)
y_train.to_csv("data/y_train.csv",header=True)
y_val.to_csv("data/y_val.csv",header=True)
word_score_series.to_csv("data/word_score_series.csv",header=True)


"""
This file exports out X_train, X_val, y_train,y_val and word_score series.
Files are exported into a foleder called "data".
Datafolder should be located in the same folder as the program file.
"""
