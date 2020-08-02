from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')


@app.route('/CAR', methods=['GET'])
def CAR():
    return render_template('CAR.html')

@app.route('/BIKE', methods=['GET'])
def BIKE():
    return render_template('BIKE.html')

#@app.route('/CAR_Result', methods=['GET','POST'])
#def car():

        

#@app.route('/BIKE_Result', methods=['GET','POST'])
#def bike():


if __name__=="__main__":
    app.run(debug=True)
