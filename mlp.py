
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import gc
import warnings

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

"""# LSTM"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

"""# Multi-Layer Perceptron"""

train_size = int(df.shape[0] * 0.67)
train_df, test_df = df.iloc[:train_size, :], df.iloc[train_size:, :]

test_df.shape

#Creating matrix dataset
def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset)-look_back):
        feature = dataset[i:i+look_back]
        target = dataset[i+1:i+look_back+1] ##### Previsão de apenas um dia
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

look_back = 4
X_train, y_train = create_dataset(train_df.values, look_back=look_back)
X_test, y_test = create_dataset(test_df.values, look_back=look_back)

import tensorflow as tf
tf.__version__

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss=root_mean_squared_error, optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=2)

train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print('Train score: {} RMSE'.format(train_score))
print('Test score: {} RMSE'.format(test_score))

df1 = data.loc[6:,:]
df1= df1.set_index(['Date'])

df1['Date'] = pd.to_datetime(data["Date"])

df1 = df1['IPCA'].str.replace(',', '.')
df1 = pd.DataFrame(df1)
df1['IPCA'] = df1['IPCA'].astype('float64')

plt.figure(figsize=(12, 8))
train_prediction = model.predict(X_train)
train_stamp = np.arange(look_back, look_back + X_train.shape[0])
test_prediction = model.predict(X_test)
test_stamp = np.arange(2 * look_back + X_train.shape[0], len(df1))
plt.plot(df1, label='true values')
plt.plot(train_stamp, train_prediction, label='train prediction')
plt.plot(test_stamp, test_prediction, label = 'test_prediction')
plt.legend();