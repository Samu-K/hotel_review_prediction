# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:37:34 2020

@author: Samu
"""
#Import for data handling
import pandas as pd
from house_rating_analysis import word_score_series
import numpy as np

#import libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from functions import draw_graph

#We define the xticks by taking the unique rounded values
x_tick = np.unique(np.array([round(num,1) for num in list(word_score_series.values)]))

#We draw a graph using the function set in the beginning
draw_graph(y=word_score_series.index,x=word_score_series.values,title="Average rating for each word",xticks=x_tick)