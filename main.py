import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



#load data
company = "FB"

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2021, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)

#prep data
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(data['Close'].values.reshape-1, 1)

predictionDays = 60

xTrain = []
yTrain = []

for x in range(predictionDays, len(scaledData)):
    xTrain.append(scaledData[x-predictionDays:x, 0])
    yTrain.append(scaledData[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

#build the model