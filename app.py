from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
from datetime import date

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')


@app.route('/Predict', methods=['GET','POST'])
def predict():
    case_data=pd.read_csv('Data/Main.csv')
    X1=case_data['serial'].iloc[:34].values
    Y1=case_data['Total Confirmed'].iloc[:34].values
    X1 = X1.reshape(-1,1)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.20,random_state=0)
    reg = LinearRegression()
    reg.fit(X_train1,Y_train1)
    y_pred1 = reg.predict(X_train1)


    X2=case_data['serial'].iloc[35:80].values
    Y2=case_data['Total Confirmed'].iloc[35:80].values
    X2 = X2.reshape(-1,1)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.20,random_state=0)
    regre = LinearRegression()
    regre.fit(X_train2,Y_train2)
    y_pred2 = regre.predict(X_train2)


    X3=case_data['serial'].iloc[81:150].values
    Y3=case_data['Total Confirmed'].iloc[81:150].values
    X3 = X3.reshape(-1,1)
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.20,random_state=0)
    regress = LinearRegression()
    regress.fit(X_train3,Y_train3)
    y_pred3 = regress.predict(X_train3)

    X4=case_data['serial'].iloc[151:].values
    Y4=case_data['Total Confirmed'].iloc[151:].values
    X4 = X4.reshape(-1,1)
    X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X4, Y4, test_size=0.20,random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train4,Y_train4)
    y_pred4 = regressor.predict(X_train4)
    #date1 = date(str(request.form["date"]))
    #def differ_days(date1, date2):
    #    a = date1
    #    b = date2
    #    return (a-b).days
    #date2=date(2020,1,30)
    #date3=differ_days(date1,date2)                 
    #return render_template('index.html',date="{}".format(date3))
    date=int(request.form["date"])
    
    if date > -1 and date < 35:
      y_pred = reg.predict([[date]])
      score=reg.score(X_test1,Y_test1)
    if date > 34 and date < 81:
      y_pred = regre.predict([[date]])
      score=regre.score(X_test2,Y_test2)
    if date > 81 and date < 151 :
      y_pred = regress.predict([[date]])
      score=regress.score(X_test3,Y_test3)
    if date > 150 :
      y_pred = regressor.predict([[date]])
      score=regressor.score(X_test4,Y_test4)
    y_pred = float(str(y_pred)[2:-2])
    score=round(score,2)
    score=score*100
    return render_template('result.html',predicted="On That Day The Number Of Corona Cases Will Be {}".format(y_pred),score="We Are {}% Sure About Our Predcition".format(score))


if __name__=="__main__":
    app.run(debug=True)
