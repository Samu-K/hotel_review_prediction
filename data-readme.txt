tripadvisor_hotel_reviews.csv:
- Contains the main data
- Freeform text reviews and rating

word_score_series_train_copy.csv:
- Contains scorings for each word
- Created in ml_data.py

X_train.csv, X_val.csv:
- Training and validation data for machine learning
- Is created in ml_data.py
- If cut is specified in ml_data.py, then this will only contain a portion of the original data
- Is encoded from categorical variables using dummies

X_train_manual, X_val_manual:
- Training and validation data for manual predicting
- Has the original text reviews (cleaned using text_cleaner) 
  and unique_words contained withing the review

y_train, y_val:
- Training and validation targets for ml
- If cut is specified in ml_data.py, then this will only contain a portion of the original data
- Contains the target and corresponding index

y_train_manual, y_val_manual:
- Training and validation targets for manual predicting
- Contains the target and corresponding index
