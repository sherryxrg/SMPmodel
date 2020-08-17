import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Input, Dense, LSTM, TimeDistributed,
                                     RepeatVector, Activation)
from pickle import dump, load


def create_model(look_back, foward_days):
    NUM_NEURONS_FirstLayer = 128
    NUM_NEURONS_SecondLayer = 64
    # Build the model
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(
        look_back, 1), return_sequences=True))
    model.add(LSTM(NUM_NEURONS_SecondLayer,
                   input_shape=(NUM_NEURONS_FirstLayer, 1)))
    model.add(Dense(foward_days))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(model, dataX, dataY, epoch_count):
    history = model.fit(dataX, dataY, batch_size=2,
                        epochs=epoch_count, shuffle=True)


def prep_data(data, input_size, output_size):
    dX, dY = [], []
    for i in range(len(data) - input_size - output_size):
        dX.append(data[i:i + input_size])
        dY.append(data[i + input_size:i + input_size + output_size])
    return dX, dY


def split_into_train_and_test(data, train_ratio):
    train_size = int(data.shape[0] * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def generate_sin_wave():
    x = np.arange(0, 10000, step=0.1)
    y = np.sin(x)
    return y


def run(name: str):
    company_name = name
    # The number of timesteps for input
    look_back = 50
    # The number of timesteps to predict
    look_foward = 50

    # Download stock prices from yahoo finance
    df = yf.Ticker(company_name).history(interval='1m', period='1wk')
    prices = df['Close'].values

    # Prepare data
    vals = prices

    # Create a new scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    vals = vals.reshape(-1, 1)
    vals = scaler.fit_transform(vals)
    x, y = prep_data(vals, look_back, look_foward)
    x = np.array(x)
    y = np.array(y)

    # Split data into test and train
    train_x, test_x = split_into_train_and_test(x, 0.75)
    train_y, test_y = split_into_train_and_test(y, 0.75)

    # Directory to save the model and scaler in
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../")
    model_dir_to_save = '{}/model_checkpoints/{}.h5'.format(
        parent_dir, company_name.lower())
    scaler_dir_to_save = '{}/model_checkpoints/{}_scaler.pkl'.format(
        parent_dir, company_name.lower())

    # Create and train a model
    model = create_model(look_back, look_foward)
    train_model(model, train_x, train_y, 1)
    model.save(model_dir_to_save)
    dump(scaler, open(scaler_dir_to_save, 'wb'))
