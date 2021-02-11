#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:21:46 2021

@author: samueljocas
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

#############Importing trained classifier and fitted vectorizer################
nb_clf = pickle.load(open("nb_clf_NYT", 'rb'))
vectorizer = pickle.load(open("vectorizer_NYT", 'rb'))

##############Predict sentiment using the trained classifier###################

# Import test data set
data_pred = pd.read_csv("/Users/samueljocas/Desktop/Data Science Portfolio/New York Times ML Algorithm  /NYT_OpEd_Test.csv", encoding = "ISO-8859-1")
X_test = data_pred.iloc[:,4] # extract column with news articl
X_vec_test = vectorizer.transform(X_test) #don't use fit_transform here because the model is already fitted
X_vec_test = X_vec_test.todense() #convert sparse matrix to dense

# Transform data by applying term frequency inverse document frequency (TF-IDF) 
tfidf = TfidfTransformer() #by default applies "l2" normalization
X_tfidf_test = tfidf.fit_transform(X_vec_test)
X_tfidf_test = X_tfidf_test.todense()

# Predict the writer of the Op-Ed Text
nyt_pred = nb_clf.predict(X_tfidf_test)
print(nyt_pred)
