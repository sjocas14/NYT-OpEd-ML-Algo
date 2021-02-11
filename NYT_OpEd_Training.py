#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:10:01 2021

@author: samueljocas
"""

# import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle

filepath = "scraped articles filepath .csv"
data = pd.read_csv(filepath, encoding = "ISO-8859-1")

X = data.iloc[:,4] # extract column with news article body

print(X)

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X.values.astype('U'))
pickle.dump(vectorizer, open("vectorizer_NYT", 'wb')) # Save vectorizer for reuse
X_vec = X_vec.todense()

tfidf = TfidfTransformer() #by default applies "l2" normalization
X_tfidf = tfidf.fit_transform(X_vec)
X_tfidf = X_tfidf.todense()


##################Apply Naive Bayes algorithm to train data####################
# Extract the news body and labels for training the classifier

X_train = X_tfidf[:,:]
Y_train = data.iloc[:,3]

# Train the NB classifier
clf = GaussianNB().fit(X_train, Y_train) 
pickle.dump(clf, open("nb_clf_NYT", 'wb')) # Save classifier for reuse
