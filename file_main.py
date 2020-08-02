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
x_data = np.array(case_data['serial'])
X = X.reshape(-1,1)
#y = np.exp(1.11292745) * np.exp(0.08338703*X_train)
Y_log=np.log(Y)
X_log=np.log(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_train)

X_log = X_log.reshape(-1,1)
X1=case_data['serial'].iloc[:34].values
Y1=case_data['Total Confirmed'].iloc[:34].values
X1 = X1.reshape(-1,1)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.20,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train1,Y_train1)
y_pred1 = regressor.predict(X_train1)


X2=case_data['serial'].iloc[35:80].values
Y2=case_data['Total Confirmed'].iloc[35:80].values
X2 = X2.reshape(-1,1)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.20,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train2,Y_train2)
y_pred2 = regressor.predict(X_train2)


X3=case_data['serial'].iloc[81:150].values
Y3=case_data['Total Confirmed'].iloc[81:150].values
X3 = X3.reshape(-1,1)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.20,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train3,Y_train3)
y_pred3 = regressor.predict(X_train3)

X4=case_data['serial'].iloc[151:].values
Y4=case_data['Total Confirmed'].iloc[151:].values
X4 = X4.reshape(-1,1)
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X4, Y4, test_size=0.20,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train4,Y_train4)
y_pred4 = regressor.predict(X_train4)

y = np.exp(1.11292745) * np.exp(0.08338703*X_train)
plt.subplot(331)
plt.scatter(X_train1, Y_train1, color='red')
plt.plot(X_train1, y_pred1, color='red')
plt.title('Date Vs Cases(Division 1)')
plt.xlabel('Date in Numbers after first case') 
plt.ylabel('Cases')
plt.legend()

plt.subplot(333)
plt.scatter(X_train2, Y_train2, color='red')
plt.plot(X_train2, y_pred2, color='blue')
plt.title('Date Vs Cases(Division 2)')
plt.xlabel('Date in Numbers after first case') 
plt.ylabel('Cases')
plt.legend()

plt.subplot(337)
plt.scatter(X_train3, Y_train3, color='red')
plt.plot(X_train3, y_pred3, color='green')
plt.title('Date Vs Cases(Division 3)')
plt.xlabel('Date in Numbers after first case') 
plt.ylabel('Cases')
plt.legend()

plt.subplot(339)
plt.scatter(X_train4, Y_train4, color='red')
plt.plot(X_train4, y_pred4, color='yellow')
#plt.plot(x_data, y,color='yellow')

plt.title('Date Vs Cases(Division 4)')
plt.xlabel('Date in Numbers after first case') 
plt.ylabel('Cases')
plt.legend()
plt.show()

while True:
 inpu=int(input("Enter"))
 y_pred1 = regressor.predict([[inpu]])
 print(y_pred1)
