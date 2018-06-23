import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_before = pd.read_csv('Google_Stock_Price_Train.csv')
training = training_before.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training = sc.fit_transform(training)

X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training[i-60:i,0])
    y_train.append(training[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping to add more dimensionalities
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#AI Layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1], 1))) #first axis taken into consideration
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(p=0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(p=0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
              
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#plotting the diagram
test_before = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_before.iloc[:,1:2].values

data_total = pd.concat((training_before['Open'],test_before['Open']), axis=0) #0 is vertical concat
inputs = data_total[len(data_total)-len(test_before)-60:].values #to numpy array
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # no need to fit anymore

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()