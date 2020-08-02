from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

case_data=pd.read_csv('Data/Main.csv')
X=case_data['serial'].values
Y=case_data['Total Confirmed'].values
X = X.reshape(-1,1)
#X = X.transpose()
Y_log=np.log(Y)
X_log=np.log(X)
X_log = X_log.reshape(-1,1)
print(np.count_nonzero(X))
print(np.count_nonzero(Y))
print(np.count_nonzero(X_log))
print(np.count_nonzero(Y_log))

Y = np.delete(Y,0)
curve_fit = np.polyfit(X_log,Y, 1)
X_log = np.array(X_log)
Y= np.array(Y)
curve_fit=np.polyfit(X_log,Y,1)
print(curve_fit)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Date')
plt.ylabel('CASES')
#plt.show()

#while True:
# inpu=int(input("Enter"))
# y_pred1 = regressor.predict([[inpu]])
# print(y_pred1+35568)
