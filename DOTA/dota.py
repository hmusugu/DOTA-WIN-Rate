# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:31:11 2018

@author: guna_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("dota.csv")

x = df.iloc[:,1:20].values
y = df.iloc[:,20].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)

df.isnull().sum()

df['Hero Damage'] = df['Hero Damage'].fillna(np.mean(df['Hero Damage']))
df['Hero Healing'] = df['Hero Healing'].fillna(np.mean(df['Hero Healing']))
df['Gold Spent'] = df['Gold Spent'].fillna(np.mean(df['Gold Spent']))
df['Tower Damage'] = df['Tower Damage'].fillna(np.mean(df['Tower Damage']))

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2 )


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
mae3 = mean_absolute_error(y_test,y_pred)
mse3 = mean_squared_error(y_test,y_pred)
Rsquared3 = r2_score(y_test,y_pred)