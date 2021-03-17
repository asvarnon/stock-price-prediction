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

model = Sequential()

model.add(LSTM(units = 50, returnSequences=True, inputShape=(xTrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, returnSequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=25, batch_size=32)

# TEST THE MODEL ACCURACY ON EXISTING DATA
testStart = dt.datetime(2020,1,1)
testEnd = dt.datetime.now()

testData = web.DataReader(company, 'yahoo', testStart, testEnd)
actualPrices = testData['Close'].values

totalDataset = pd.concat((data['Close']. testData['Close']), axis=0)
modelInputs = totalDataset[len(totalDataset) - len(testData) - predictionDays:].values
modelInputs = modelInputs.reshape(-1,1)
modelInputs = scaler.transform(modelInputs)

#make predictions on test data

for x in range(predictionDays, len(modelInputs)):
    xTest.append(modelInputs[x-predictionDays:x, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

predictedPrices = model.predict(xTest)
predictedPrices = scaler.inverse_transform(predictedPrices)


#plot test predictions

plt.plot(actualPrices, color='black', label=f'Actual {company} price')
plt.plot(predictedPrices, color='green', label=f'predicted {company} price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()
