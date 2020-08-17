"""
Code structure/functions: Sherry Guo
Model reference: Yacoub (https://tinyurl.com/rsr597n)

[ Glossary ]
model.compile -- build model architecture with optimizer
model.fit -- train model (model learns by adjusting weights)
model.predict -- difference from fit is that it doesn't have input targets
"""
from utils import *
from utils_2 import *
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import models
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')
np.set_printoptions(precision=2, suppress=True)

# -- constants
DATE_START = '2020-01-01'
DATE_END = '2020-08-01'
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

scaled_data, train_data, test_data = \
                                tech_prep_data(dataset=dataset,
                                               train_size=TRAIN_SIZE,
                                               hist_days=HIST_DAYS,
                                               scaler=scaler)

x_train_nd, y_train, scaled_tech_ind = tech_generate_sets(hist_days=HIST_DAYS,
                                         num_columns=NUM_COL,
                                         input_data=train_data,
                                         scaler=scaler)

model_name = "model" + STOCK_NAME + "_8-16" + ".h5"

# todo: !! uncomment to re-compile, fits and exports model
tech_fit_model(x_train_nd, y_train, tech_train=scaled_tech_ind, model_name=model_name)

# create_model_jihyo(x_train_nd, y_train, model_name=model_name)
model = models.load_model(model_name)

# ===========================================================
#                       TESTING
# ===========================================================

x_test_nd, y_test = generate_sets(hist_days=HIST_DAYS,
                                  num_columns=NUM_COL,
                                  input_data=test_data)


# ===========================================================
#               Predicting w/ imputation
# ===========================================================

holder_list = np.zeros((0, 5))

# todo: !! try setting future_days to different numbers to see
#  more or less predictions
predicted_list = tech_get_data_wpredicted(future_days=5,
                                     dataset=dataset,
                                     model=model,
                                     scaler=scaler,
                                     num_columns=NUM_COL,
                                     predicted_list=holder_list)

print(predicted_list)

print("====== actual data =======")
dataset = get_data(stock_name=STOCK_NAME,
                   data_source=DATA_SOURCE,
                   start_date='2020-08-01',
                   end_date='2020-08-06')


# ===========================================================
#                       Graph metrics
# ===========================================================
# history = pickle.load(open('metrics/hist_modelAAPL_2', 'rb'))
# print(history)
#
# get_plot(history, 'mae')




















