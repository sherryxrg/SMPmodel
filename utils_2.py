"""
Functions for importing data and getting training sets
(Tech-ind model)
"""
import pickle
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras import backend
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation, concatenate
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def tech_get_data(stock_name, data_source, start_date, end_date):
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
    print("=== summary ===")
    print(df.head(), "\n")
    print(f"Columns: {num_col},  Rows: {num_rows}")

    return dataset


def tech_prep_data(dataset, train_size, hist_days, scaler):
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

    # technical_indicators = []
    # tech_sets = np.zeros((0, dataset.shape[1]))
    # tech_sets_nd = np.zeros(((len(dataset) - hist_days), hist_days, dataset.shape[1]))
    #
    # # chop into {hist_days} chunks and find average of each chunk (close)
    # # append to technical indicators list
    # for i in range(hist_days, len(dataset)):
    #     tech_sets = dataset[i - hist_days:i, 0:dataset.shape[1]]
    #     tech_sets_nd[i - hist_days, :, :] = tech_sets
    #
    # for set in tech_sets_nd:
    #     sma = np.mean(set[:, 3])
    #     technical_indicators.append(np.array([sma]))
    #
    # technical_indicators = np.array(technical_indicators)
    # scaled_tech_ind = scaler.fit_transform(technical_indicators)
    #
    # train_tech_ind = scaled_tech_ind[0:train_len, :]
    # test_tech_ind = scaled_tech_ind[train_len - hist_days:, :]
    #
    # print("tech sets: ", technical_indicators)

    # scale before splitting into train/test
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:train_len, :]
    print("\n=== train data ===")
    print("shape: ", train_data.shape)
    print("first two rows: ")
    print(train_data[0:2])

    test_data = scaled_data[train_len - hist_days:, :]
    print("\n=== test data ===")
    print("shape: ", test_data.shape)
    print("first two rows: ")
    print(test_data[0:2])

    return scaled_data, train_data, test_data


def tech_generate_sets(hist_days, num_columns, input_data, scaler):
    """
    Generates the datasets used for training or testing,
    and also technical indicators.
    training: x_set = x_train
    testing: x_set = x_test

    :param input_data: subset data used for train or test
    :param hist_days: int, number of days the predicted point learns from
    :param num_columns: int, must match number of cols imported
    :return: x_set_nd, y_set
    """
    x_set = np.zeros((0, num_columns))
    y_set = np.zeros((0, num_columns))
    x_set_nd = np.zeros(((len(input_data) - hist_days), hist_days, num_columns))

    for i in range(hist_days, len(input_data)):
        x_set = input_data[i - hist_days:i, 0:num_columns]
        x_set_nd[i - hist_days, :, :] = x_set
        y_set = np.vstack((y_set, input_data[i, 0:num_columns]))

    # make tech indicators
    technical_indicators = []
    tech_sets = np.zeros((0, input_data.shape[1]))
    tech_sets_nd = np.zeros(
        ((len(input_data) - hist_days), hist_days, input_data.shape[1]))

    # chop into {hist_days} chunks and find average of each chunk (close)
    # append to technical indicators list
    for i in range(hist_days, len(input_data)):
        tech_sets = input_data[i - hist_days:i, 0:input_data.shape[1]]
        tech_sets_nd[i - hist_days, :, :] = tech_sets

    for set in tech_sets_nd:
        sma = np.mean(set[:, 3])
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)
    scaled_tech_ind = scaler.fit_transform(technical_indicators)

    print("tech sets: ", technical_indicators)

    print("\n=*=*=*=*=* SUMMARY *=*=*=*=*=*=*")
    print("final shape of x_set: ", x_set_nd.shape, type(x_set_nd))
    print("final shape of y_set: ", y_set.shape, type(y_set))
    print("final shape of scaled_tech: ", scaled_tech_ind.shape)

    return x_set_nd, y_set, scaled_tech_ind


def tech_fit_model(x_set_nd, y_set, tech_train, model_name=None):
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
    # define two sets of inputs
    lstm_input = Input(shape=(x_set_nd.shape[1], 5), name='lstm_input')
    dense_input = Input(shape=(tech_train.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate(
        [lstm_branch.output, technical_indicators_branch.output],
        name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(5, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and then output a single value
    model = Model(
        inputs=[lstm_branch.input, technical_indicators_branch.input],
        outputs=z)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam,
                  loss='mse')

    history = model.fit(x=[x_set_nd, tech_train], y=y_set, batch_size=32,
                        epochs=50, shuffle=True, validation_split=0.1)

    # save history to print metrics later:
    history_name = "metrics/hist_" + model_name[:-3]
    with open(history_name, 'wb') as f:
        pickle.dump(history.history, f)

    if model_name is None:
        model.save('model_default.h5')
    else:
        model.save(model_name)


def tech_get_data_wpredicted(future_days, dataset, model, scaler, num_columns,
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
        return tech_get_data_wpredicted(future_days-1, dataset, model, scaler,
                                   num_columns, predicted_list)