"""
Code structure/functions: Sherry Guo
Model reference: Yacoub (https://tinyurl.com/rsr597n)

[ Glossary ]
model.compile -- build model architecture with optimizer
model.fit -- train model (model learns by adjusting weights)
model.predict -- difference from fit is that it doesn't have input targets

"""
from .utils.py import *
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras import optimizers
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# -- constants
DATE_START = '2018-01-01'
DATE_END = '2020-06-01'
STOCK_NAME = 'AAPL'
DATA_SOURCE = 'yahoo'

TRAIN_SIZE = 0.8  # train-test split ratio
HIST_DAYS = 60  # days of past data to use
NUM_COL = 5  # number of data columns (e.g. 'Open', 'Close')

# ---- import data
df = web.DataReader(STOCK_NAME, data_source=DATA_SOURCE,
                    start=DATE_START, end=DATE_END)
print("[** preview of import **]")
pd.set_option('display.max_columns', 1000)
print(df.head(), "\n")
print("shape: ", df.shape)

# ---- graph initial df

# ---- extract df columns, convert df to numpy array
data = df.filter(['High', 'Low', 'Open', 'Close', 'Volume'])
dataset = data.values
num_col = dataset.shape[1]
num_rows = dataset.shape[0]
print(f"Columns: {num_col},  Rows: {num_rows}")

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

# ---- Model
# architecture
lstm_input = Input(shape=(x_train_nd.shape[1], 5), name='input_lstm')
x = LSTM(50, name='lstm_1')(lstm_input)
x = Dropout(0.2, name='lstm_dp_1')(x)
x = Dense(64, name='dense_1')(x)
x = Activation('sigmoid', name='active_sig_1')(x)
x = Dense(NUM_COL, name='dense_2')(x)
output = Activation('linear', name='output_linear')(x)

# compile with optimizer
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')  # opt: add metrics
# opt: add validation_split, shuffle=True
hist = model.fit(x=x_train_nd, y=y_train, batch_size=32, epochs=50)

# plot model loss...add other metrics
plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()

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





















