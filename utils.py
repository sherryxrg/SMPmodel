"""
Functions for importing data and getting training sets
"""
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def generate_sets(hist_days, num_columns, input_data):
    """
    Generates the datasets used for training or testing.
    training: x_set = x_train
    testing: x_set = x_test

    :param input_data: subset data used for train or test
    :param hist_days: int, number of days the predicted point learns from
    :param num_columns: int, must match number of cols imported
    :return: x_set, y_set
    """
    x_set = np.zeros((0, num_columns))
    y_set = np.zeros((0, num_columns))
    x_set_nd = np.zeros(((len(input_data) - hist_days), hist_days, num_columns))

    for i in range(hist_days, len(input_data)):
        x_set = input_data[i - hist_days:i, 0:num_columns]
        x_set_nd[i - hist_days, :, :] = x_set
        y_set = np.vstack((y_set, input_data[i, 0:num_columns]))

    print("\n=*=*=*=*=* SUMMARY *=*=*=*=*=*=*")
    print("final shape of x_set: ", x_set_nd.shape, type(x_set_nd))
    print("final shape of y_set: ", y_set.shape, type(y_set))

    return x_set_nd, y_set


def prep_data(dataset, train_size, hist_days, scaler):
    """
    Splits and scales the data for generating training sets.
    MinMaxScaler with this default range scales all datapoints
    within 1 and 0 so big numbers don't interfere with learning.

    :param hist_days: int, number of days the predicted point learns from
    :param dataset: entire dataset
    :param scaler: type of scale applied to data
    :param train_size:
    :return: scaled_data - the scaled, full-sized dataset
             train_len - the length of the training set
             train_data - subset of scaled data used for training
    """
    train_len = math.ceil(len(dataset) * train_size)
    print("train set length: ", train_len)

    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:train_len, :]
    print("\n=== train data ===")
    print("shape: ", train_data.shape)
    print("first two rows: ")
    print(train_data[0:2])

    test_data = scaled_data[train_len - hist_days:, :]
    print("\n=== test data ===")
    print("shape: ", train_data.shape)
    print("first two rows: ")
    print(test_data[0:2])

    return scaled_data, train_data, test_data


def get_data_wpredicted(future_days, dataset, model, scaler, num_columns,
                        predicted_list):
    """
    Similar to get_data, except this also contains imputed future data
    that does not yet exist.

    :param model: model used for prediction
    :param num_columns: number of data columns
    :param predicted_list: growing list of predictions
    :param scaler: used to scale dataset for predicting
    :param future_days: how many days into the future to predict for
    :param dataset: String (e.g. AAPL)
    :return: predicted_list
    """
    # dataset + predicted rows from previous calls
    current_data = np.concatenate((dataset, predicted_list))
    # only take the most recent 60 entries
    current_data = current_data[-60:]

    # base case
    if future_days == 0:
        return predicted_list
    else:
        # do round of predictions
        last_xdays_scaled = scaler.transform(current_data)
        x_test = [last_xdays_scaled]
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_columns))
        pred_price = model.predict(x_test)
        pred_price = scaler.inverse_transform(pred_price)
        predicted_list = np.concatenate((predicted_list, pred_price))

        # append results to predicted list
        return get_data_wpredicted(future_days-1, dataset, model, scaler,
                                   num_columns, predicted_list)


def get_data(stock_name, data_source, start_date, end_date):
    """
    Gets data from web and converts it to numpy array.

    :param stock_name: String (e.g. AAPL)
    :param data_source: String (e.g. Yahoo)
    :param start_date: String (e.g. 2012-01-01)
    :param end_date: String
    :return: dataset, a numpy array
    """
    print("getting data...")
    df = web.DataReader(stock_name, data_source=data_source,
                        start=start_date, end=end_date)
    pd.set_option('display.max_columns', 1000)
    data = df.filter(['High', 'Low', 'Open', 'Close', 'Volume'])
    dataset = data.values
    num_col = dataset.shape[1]
    num_rows = dataset.shape[0]
    print("=== summary ==")
    print(df.head(), "\n")
    print(f"Columns: {num_col},  Rows: {num_rows}")

    return dataset


def fit_model(x_train, y_train, model_name=None):
    """
    Compiles and fits the model and saves it as a h5 file.
    Also generates metrics generated from training.

    NOTE: running without own model_name will override current model,
    and your prediciton numbers may change.

    :param x_train: numpy array (input of shape (x, y, z))
    :param y_train: numpy array (prediction labels of shape (x, z))
    :param model_name: String
    :return: hist a history callback including metrics
    """
    # architecture
    lstm_input = Input(shape=(x_train.shape[1], x_train.shape[2]), name='input_lstm')
    x = LSTM(50, name='lstm_1')(lstm_input)
    x = Dropout(0.2, name='lstm_dp_1')(x)
    x = Dense(64, name='dense_1')(x)
    x = Activation('sigmoid', name='active_sig_1')(x)
    x = Dense(x_train.shape[2], name='dense_2')(x)
    output = Activation('linear', name='output_linear')(x)

    # compile with optimizer
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')  # opt: add metrics
    # opt: add validation_split, shuffle=True
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=50)

    if model_name is None:
        model.save('model_default.h5')
    else:
        model.save(model_name)

    # generate model plot
    _get_plots(history)


def _get_plots(history):
    """
    Plots the loss history as a graph.
    Loss shrinks as model learns from the data.

    :param history:
    :return:
    """
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('metrics_model_loss.png')

