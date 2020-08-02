from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

case_data=pd.read_csv('Data/Main.csv')
X=case_data['serial'].values
Y=case_data['Total Confirmed'].values
X = X.reshape(-1,1)
#X = X.transpose()
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
while True:
 inpu=int(input("Enter"))
 y_pred1 = regressor.predict([[inpu]])
 print(y_pred1)
