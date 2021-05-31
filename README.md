# hotel_review_prediction
The goal of this project is to use data from written hotel reviews to build a pipeline to handle given data and use a model to predict what rating on a scale of 0-5 that rating is.

functions.py contains all functions used throughout the project \
ml_data.py cleans, handles and sorts the main data. \
manual_prediciton.py uses a manually made way of predicting the review. \
(While fairly accurate, this was simply an experiment I wanted to try)
ml_prediction.py use an XGBRegressor to attempt to predict review scores.

The main data is reviews scraped from TripAdvisor. \
The data has been retrieved using Kaggle. \
Link here: https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

This project was my first attempt at cleaning and predicting from text. It's purpose was to learn language processing and building a pipeline.
