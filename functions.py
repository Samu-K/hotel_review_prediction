# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:23:15 2020

@author: Samu
"""
#For text handling
import regex as re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Make temp to allow function settings
word_score_series_copy = ""
word_score = {}

#We'll make a few functions to make code cleaner

def text_cleaner(text):
    """
    Cleans up text paragraphs.
    Takes ous dots and commas.
    Replaces won't with will not, isn't with is not and didn't with did not
    Takes out any numbers or non-writing characters
    """
    text = text.lower()
    text = re.sub('\.',' ',text)
    text = re.sub('\,',' ',text)
    text = re.sub("won't","will not",text)
    text = re.sub("isn't","is not",text)
    text = re.sub("didn't","did not",text)
    text = re.sub("\d",'',text)
    text = re.sub("\W",' ',text)
    text = re.sub("_",'',text)
    text = " ".join(text.split())
    
    return text

def unique_words(text):
    """
    Take the unique words in a text paragraph.
    """
    all_words = text.split()
    
    #Take out words that are less than 2 characters
    accepted_words = [word for word in all_words if len(word) > 2]
    
    #Turn the list into an array to use numpys unique function
    unique_words = np.unique(np.array(accepted_words))
    
    return unique_words

def unique_word_count(text,unique_words,all_words=""):
    """
    The the amount of times a word shows up in a text paragraph.
    """
    
    word_count = {}
    #The function can be supplied the words
    #If not, we split the given text here
    if all_words == "":
        all_words = text.split()
    
    #loop through every word in the review
    for word in all_words:
        #See if that word is in our accepted words
        if word in unique_words:
            #Check if we already have an entry for this word
            if word in word_count.keys():
                #We just add another instance
                word_count[word].append(word)
            else:
                #If not we make a new entry
                word_count[word] = [word]
        else:
            #If not in accepted words, we just skip
            continue
    
    #We turn the dict into a list, and take the lenght of each list, 
    #which correspons to the amount of times the word showed up
    word_count = pd.Series(word_count).apply(len)
    
    return word_count
        
def word_scorer(df,target_series,index):
    """
    Loop through the index of the supplied dataframe,
    and apply the target value for each word
    """
    words = df.unique_words[index]
    rating = target_series[index]
    
    return words, rating
    
def average_rating(x,min_len=1000):
    """
    Takes a list. and returns the average if the length is more than min_len.
    It neither match, it returns na
    """
    l = len(x)
    avg = np.average(x)
    if l > min_len:
        return(avg)
    else:
        return(None)

def review_predictor(word_values,words,word_count):
    """
    Predict the rating of the review based on the words given
    """
    total_ratings = []
    #Go through every word in the review
    for word in words:
        #Set this up for a while loop later
        x=0
        #Get how many times the word showed up in the interview
        count = word_count[word]
        
        #See if the word is in our values list
        
        if word in word_values:
            #if it is we fetch the value
            word_rating = word_values[word]
            
        else:
            #If not we restart the loop
            continue
        
        #This adds the word rating to the list as many times as the word showed up in the original review
        if count != 0:
            while x<count:
                total_ratings.append(float(word_rating))
                x+=1
        else:
            total_ratings.append(float(word_rating))
    
    #Get the average rating
    if len(total_ratings) == 0:
        return None
    else:
        pred = sum(total_ratings) / len(total_ratings)
        return pred

def predict(text,word_values):
    """
    Combine variable creating, and prediction
    """
    #values assigned to each word
    text_clean = text_cleaner(text)
    words = unique_words(text_clean)
    word_count = unique_word_count(text_clean,words)
    
    pred = review_predictor(word_values,words,word_count)
    return pred

def unrated_value_drop(data,word_scores):
    word_list = data
    for word in word_list:
        
        if word not in word_scores:
            word_list.remove(word)
        else:
            pass
    
    return word_list

def draw_graph(y,x,figsize=(40,20),yticks=[],xticks=[0,1,2,3,4,5],palette="YlGnBu",title="",orient='h'):
    """
    Draw a barplot, with the given values
    """
    #Set the figure big enough for our data
    plt.figure(figsize=figsize)
    plt.title(title,fontsize=18)

    #Make a stacked barplot with the word on the y-axis and the word rating as the x-asis
    sns.barplot(y=y,x=x,orient=orient)
    sns.color_palette(palette)
    #Hide the labels for the words (unreadable on this scale)
    plt.yticks(yticks)
    #Set the review scale manually
    plt.xticks(xticks)
    plt.show()