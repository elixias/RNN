import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training = pd.read_csv('Google_Stock_Price_Train.csv').iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScalar(feature_range=(0,1))
training = sc.fit_transform(training)

X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training[i-60:i,0])
    y_train.append(training[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping to add more dimensionalities
X_train = np.reshape(X_train, order=(X_train.shape[0],X_train.shape[1], 1))

#AI Layers
from keras.model import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1], 1))) #first axis taken into consideration
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50,return_sequences=True)
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50,return_sequences=True)
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50)
regressor.add(Dropout(p=0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_square_error')