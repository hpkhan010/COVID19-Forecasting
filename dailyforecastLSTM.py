import numpy as np
from numpy import array
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, LSTM, BatchNormalization, 
                                    Bidirectional, TimeDistributed)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from statsmodels.tsa.stattools import adfuller


# split a univariate sequence into samples

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Train-Test Split

def traintestSplit(data, split):
	global train, test
	size = int(len(data.head(-split)))
	train, test = data[0:size], data[size : len(data)]
	print(train.shape)
	print(test.shape)
	return train, test


DATE = '05-15-2020'
NUM_STEPS = 7
NUM_FEATURES = 1
NUM_UNITS = 464
NUM_EPOCHS = 150
NUM_BATCH = 16
DROPOUT_RATE = 0.05

# Daily Cases from https://ourworldindata.org/grapher/daily-cases-covid-19

dailyCases = pd.read_csv("D:\Projects\covid19\daily-cases-covid-19.csv")
df = pd.DataFrame(dailyCases)
df = df.loc[df["Entity"] == "United States"]
dt_index = pd.to_datetime(df["Date"])
df.index = dt_index
df = df.iloc[:, 3:]
df.rename(columns={df.columns[0]: "Actual"}, inplace=True)
df = pd.DataFrame(df)

# Dropping first few months when daily reported cases were very low

cutoff = pd.to_datetime('2020-03-20')
df = df.loc[(df.index > cutoff)]
plot = plt.plot(df["Actual"], "ob-", label="Actual")
plt.title("Daily Cases")
plt.legend(loc="upper left", fontsize=8)
plt.show()
df.to_csv("D:\Projects\covid19\daily_cases_US.csv")

# Checking for stationarity

X = df["Actual"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


traintestSplit(df, 14)
scaler = MinMaxScaler(feature_range=(0, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)
X_train, y_train = split_sequence(train_sc, NUM_STEPS)
X_test, y_test = split_sequence(test_sc, NUM_STEPS)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], NUM_FEATURES))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], NUM_FEATURES))


K.clear_session()
model = Sequential()
model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=False), 
						input_shape=(NUM_STEPS, NUM_FEATURES)))
model.add(Dense(NUM_UNITS, activation='relu'))                        
model.add(Dense(1))
model.add(Dropout(0.05))
model.compile(optimizer='adam', loss='mse', metrics = ['acc'])
model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=NUM_BATCH)


yhat = model.predict(X_test, verbose=0)
scaler = MinMaxScaler(feature_range=(0, 1))
obj = scaler.fit(train)
y_pred = obj.inverse_transform(yhat)
y_actual = obj.inverse_transform(y_test)
r2_test = r2_score(y_actual, y_pred)
print("R-squared is: %f"%r2_test)
rmse = sqrt(mean_squared_error(y_actual, y_pred))
print("RMSE is: %f"%rmse)
plt.plot(y_actual, color = 'green')
plt.plot(y_pred, color = 'red')
MAPE = np.mean(np.abs(y_pred - y_actual) / np.abs(y_actual))  # MAPE
accuracy = 1 - MAPE
print("Model Accuracy: " + "{:.2%}".format(accuracy))

pred_list = []
batch = train_sc[-NUM_STEPS:].reshape((1, NUM_STEPS, NUM_FEATURES))

for i in range(NUM_STEPS):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-NUM_STEPS:].index, columns=['Prediction'])

df_test = pd.concat([df,df_predict], axis=1)

plt.figure(figsize=(20, 5))
plt.plot(df_test.index, df_test['Actual'])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()