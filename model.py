"""
Code structure/functions: Sherry Guo
Model reference: Yacoub (https://tinyurl.com/rsr597n)

[ Glossary ]
model.compile -- build model architecture with optimizer
model.fit -- train model (model learns by adjusting weights)
model.predict -- difference from fit is that it doesn't have input targets

"""
from utils import *
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import models
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')


# -- constants
DATE_START = '2018-01-01'
DATE_END = '2020-06-01'
STOCK_NAME = 'AAPL'
DATA_SOURCE = 'yahoo'

TRAIN_SIZE = 0.8  # train-test split ratio
HIST_DAYS = 60  # days of past data to use
NUM_COL = 5  # number of data columns (e.g. 'Open', 'Close')

# get data
dataset = get_data(STOCK_NAME, DATA_SOURCE, DATE_START, DATE_END)

# ===========================================================
#                       TRAINING
# ===========================================================

# ---- prep data
train_len = math.ceil(len(dataset) * TRAIN_SIZE)
print("train set length: ", train_len)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:train_len, :]
print("\n[** train data **]")
print("shape: ", train_data.shape)
print(train_data[0:2])
print()

# populate training set for model
x_train = np.zeros((0, NUM_COL))
y_train = np.zeros((0, NUM_COL))
x_train_nd = np.zeros(((len(train_data)-HIST_DAYS), HIST_DAYS, NUM_COL))

for i in range(HIST_DAYS, len(train_data)):
    x_train = train_data[i-HIST_DAYS:i, 0:NUM_COL]
    x_train_nd[i-HIST_DAYS, :, :] = x_train
    y_train = np.vstack((y_train, train_data[i, 0:NUM_COL]))
    # make sure the test sets are stored correctly
    if i <= (HIST_DAYS+1):
        print("preview of a unit of x_train: ")
        print(x_train.shape)
        # print(x_train)
        # print("\npreview of a unit of y_train: ")
        # print(y_train)

print("\n=*=*=*=*=* TRAIN SET SUMMARY *=*=*=*=*=*=*")
print("final shape of x_train: ", x_train_nd.shape, type(x_train_nd))
print("final shape of y_train: ", y_train.shape, type(y_train))


# re-compile, fits and exports model
# fit_model(x_train_nd, y_train)
model = models.load_model('model_default.h5')

# ===========================================================
#                       TESTING
# ===========================================================

np.set_printoptions(precision=2, suppress=True)

test_data = scaled_data[train_len - HIST_DAYS:, :]
x_test = np.zeros((0, NUM_COL))
y_test = np.zeros((0, NUM_COL))  # 2D
x_test_nd = np.zeros(((len(test_data)-HIST_DAYS), HIST_DAYS, NUM_COL))  # 3D

# x_test is the 3D array of {HIST_DAYS} used for training
# y_test are the labels to x_test checks against
for i in range(60, len(test_data)):
    x_test = test_data[i - HIST_DAYS:i, 0:NUM_COL]
    x_test_nd[i-HIST_DAYS, :, :] = x_test
    y_test = np.vstack((y_test, test_data[i, 0:NUM_COL]))

print("shape of x_test_nd:", x_test_nd.shape)
print("shape of y_test:", y_test.shape)

predictions = model.predict(x_test_nd)
predictions = scaler.inverse_transform(predictions)  # revert to original scale
print("predictions \n", predictions[0])
y_test_check = scaler.inverse_transform(y_test)
print("actual \n", y_test_check[0])

quote = web.DataReader(STOCK_NAME, data_source=DATA_SOURCE,
                       start='2018-01-01', end='2020-06-01')

new_df = quote.filter(['High', 'Low', 'Open', 'Close', 'Volume'])
last_xdays = new_df[-HIST_DAYS:].values
# todo: find a way to continually use past days prediction as data

last_xdays_scaled = scaler.transform(last_xdays)
X_test = [last_xdays_scaled]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("predicted prices:\n", pred_price)





















