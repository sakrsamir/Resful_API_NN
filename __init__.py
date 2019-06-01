from flask import render_template, jsonify, Flask
import pandas
import numpy as np
import os, json
#import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K 

app = Flask(__name__)

def getData(dayNumber, path, features, timeStep):
    data = pandas.read_csv(path)
    data = data.get(features).values
    fristIndex = len(data) - (dayNumber + timeStep)
    data = data[fristIndex : ].reshape(-1,1)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler = scaler.fit(data)
    data = scaledData(data, scaler)
    xInput = np.array([ data[i : i + timeStep] for i in range(1, len(data) - timeStep + 1)])
    return xInput, scaler

def predictModel(model, data):
    y_pred = model.predict(data)
    return y_pred

def loadModel(path):
    model = load_model(path)
    return model

def scaledData(data, scaler, normalize = 'normalize'):
    if normalize == 'normalize':
        scaledData = scaler.fit_transform(data)
        return scaledData
    elif normalize == 'invert':
        scaledData = scaler.inverse_transform(data)
        return scaledData


@app.route('/')
@app.route('/Documentation')
def home():
	return '<center><h1>Resful API to get predict of our NN model backend tensorflow & keras</h1<br><h3>author : Sakr Samir</h3></center>'

@app.route('/pre_nn')
def NN():
	x, scaler = getData(100, 'FB-7.csv', 'Close', 60)

	model = loadModel('27042019-135806-LSTM.h5')

	pre = predictModel(model, x)
	pre = scaledData(pre, scaler, 'invert')
	x = str(pre[99])
	K.clear_session()
	return jsonify(predict=x)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)