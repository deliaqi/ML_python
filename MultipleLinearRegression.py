#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:42:49 2017

@author: liujiaqi
"""


import pandas as pd

# Importing the dataset
df = pd.read_csv('/Users/liujiaqi/Git/ML_python/data/IFPUG4.csv')

# Remove all rows that have any NaN values
newdf = df.dropna(axis=0,how='any')
newdf = newdf.drop('Recording Method',1)
newdf = newdf.drop('DBMS Used',1)


y = newdf.iloc[:,8].values


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Transform Label data to numeric data
labelencoder = LabelEncoder()
newdf['Development Type'] = labelencoder.fit_transform(newdf['Development Type'])
newdf['Development Platform'] = labelencoder.fit_transform(newdf['Development Platform'])
newdf['Language Type'] = labelencoder.fit_transform(newdf['Language Type'])
newdf['Organisation type'] = labelencoder.fit_transform(newdf['Organisation type'])
newdf['How Methodology Acquired'] = labelencoder.fit_transform(newdf['How Methodology Acquired'])

# OneHotEncoder Transform
X = newdf.iloc[:, :-1].values
onehotencoder = OneHotEncoder(categorical_features = [2,3,4,5,6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(np.arange(25),y_test, color = 'red',label='y_test')
plt.scatter(np.arange(25),y_pred, color = 'blue',label='y_pred')
plt.legend(loc=2);
plt.show()


