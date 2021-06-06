#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, precision_score


#%% Set the path

print("current directory is : " + os.getcwd())
os.chdir(f'{os.getenv("HOME")}/documents/spring/b/social media/final project')
print("current directory is : " + os.getcwd())

#%% Generate the data for model

mystop = set(stopwords.words("english"))|set(stopwords.words("spanish"))|set(stopwords.words("portuguese"))|set(stopwords.words("french"))|set(stopwords.words("german"))

data1 = pd.read_csv("noncompliant.txt", header= None)
data1.columns = ['tweets']
data1["type"] = 1

data2 = pd.read_csv("compliant.txt", header= None)
data2.columns = ['tweets']
data2["type"] = 0

data = data1.append(pd.DataFrame(data = data2), ignore_index=True)

tweets = [re.sub("[^a-zA-Z]", " ", x.lower()) for x in data['tweets'].tolist()]
s = ' '.join( tweets )
vocab = list( set( s.split() ) - mystop ) 
vectorizer = CountVectorizer( vocabulary=vocab  )
X = vectorizer.fit_transform( tweets ).toarray()

y = data['type']== 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)


#%% Build and train the model

clf = svm.SVC()
scores = cross_val_score(clf, X_train, y_train, cv=4)
scores.mean()

clf = svm.SVC(probability=True, kernel='rbf', class_weight="balanced", C=2.5)
clf.fit( X_train, y_train )


y_pred = clf.predict( X_test )           
confusion_matrix( y_test, y_pred )
accuracy_score( y_test, y_pred )
f1_score( y_test, y_pred )
precision_score( y_test, y_pred ) 
recall_score( y_test, y_pred )
y_prob = clf.predict_proba( X_test )[:, 1]
roc_auc_score( y_test, y_prob )


### Use cross-validation and Grid- search to find the best model with the highest accuracy score

kfolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
parameters = {'kernel':('rbf', 'sigmoid'), 'C':[2.5]}
grid_svm = GridSearchCV( svm.SVC(probability=True, class_weight="balanced"), parameters, cv=kfolds, scoring="roc_auc", verbose=1, n_jobs=-1) 
grid_svm.fit(X_train, y_train)
grid_svm.best_params_
## {'C': 2.5, 'kernel': 'rbf'}

grid_svm.best_score_
## 0.7974813633329633

grid_svm.score(X_test, y_test)
## 0.832945358063628

y_pred = grid_svm.predict( X_test ) 
y_prob = grid_svm.predict_proba( X_test )[:, 1] 
roc_auc_score( y_test, y_prob )
## 0.832870266075943

accuracy_score( y_test, y_pred )
## 0.77

f1_score( y_test, y_pred )
## 0.7809523809523811

precision_score( y_test, y_pred ) 
## 0.7699530516431925

recall_score( y_test, y_pred )
## 0.7922705314009661

#%% Apply the trained model to the csv file

## import the data
df = pd.read_csv("customertweets.csv", header= None)
df.columns = ['index','tweets']

## convert the format of the tweets
tweets = [re.sub("[^a-zA-Z]", " ", x.lower()) for x in df['tweets'].tolist()]
content = vectorizer.transform( tweets ).toarray()

clf = svm.SVC(probability=True, kernel='rbf', class_weight="balanced", C=2.5)
y_prob = clf.predict_proba( content )[:, 1]

df["Probability"] = y_prob

data = df.loc[df.iloc[:,2] > 0.968]
data1 = data.iloc[:,0:2]

data1.to_csv('Noncompliant.csv', index=False)


df = pd.read_csv("Xue_Wang.csv", header= None)






