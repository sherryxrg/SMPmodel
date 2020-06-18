"""
Code structure/functions: Sherry Guo
Model reference: Yacoub (https://tinyurl.com/rsr597n)

[ Glossary ]
model.compile -- build model architecture with optimizer
model.fit -- train model (model learns by adjusting weights)
model.predict -- difference from fit is that it doesn't have input targets
"""
from utils import *
import pandas_datareader as web
import numpy as np
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

# ===========================================================
#                       TRAINING
# ===========================================================
# get data
dataset = get_data(stock_name=STOCK_NAME,
                   data_source=DATA_SOURCE,
                   start_date=DATE_START,
                   end_date=DATE_END)

# scale all data (test set inclusive)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(dataset)

scaled_data, train_data, test_data = prep_data(dataset=dataset,
                                               train_size=TRAIN_SIZE,
                                               hist_days=HIST_DAYS,
                                               scaler=scaler)

x_train_nd, y_train = generate_sets(hist_days=HIST_DAYS,
                                    num_columns=NUM_COL,
                                    input_data=train_data)

# todo: !! uncomment to re-compile, fits and exports model
# fit_model(x_train_nd, y_train, model_name=None)
model = models.load_model('model_default.h5')

# ===========================================================
#                       TESTING
# ===========================================================
np.set_printoptions(precision=2, suppress=True)

x_test_nd, y_test = generate_sets(hist_days=HIST_DAYS,
                                  num_columns=NUM_COL,
                                  input_data=test_data)

# ---
predictions = model.predict(x_test_nd)
predictions = scaler.inverse_transform(predictions)  # revert to original scale
print("*-- check first row of predictions to actual:")
print("predictions \n", predictions[0])
y_test_check = scaler.inverse_transform(y_test)
print("actual \n", y_test_check[0])


# ===========================================================
#               Predicting w/ imputation
# ===========================================================

holder_list = np.zeros((0, 5))

predicted_list = get_data_wpredicted(future_days=3,
                                     dataset=dataset,
                                     model=model,
                                     scaler=scaler,
                                     num_columns=NUM_COL,
                                     predicted_list=holder_list)

print(predicted_list)

# quote = web.DataReader(STOCK_NAME, data_source=DATA_SOURCE,
#                        start='2018-01-01', end='2020-06-01')
#
# new_df = quote.filter(['High', 'Low', 'Open', 'Close', 'Volume'])
# last_xdays = new_df[-HIST_DAYS:].values
# # todo: find a way to continually use past days prediction as data
#
# last_xdays_scaled = scaler.transform(last_xdays)
# X_test = [last_xdays_scaled]
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
# pred_price = model.predict(X_test)
# pred_price = scaler.inverse_transform(pred_price)
# print("final predicted prices:\n", pred_price)





















