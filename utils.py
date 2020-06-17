"""
Functions for importing data and getting training sets
"""
import pandas_datareader as web
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras import optimizers

# constants
TRAIN_SIZE = 0.8  # train-test split ratio
HIST_DAYS = 60  # days of past data to use
NUM_COL = 5  # number of data columns (e.g. 'Open', 'Close')


def fit_model(x_train, y_train, model_name=None):
    """
    Compiles and fits the model and saves it as a h5 file.

    NOTE: run this function ONLY if you want to override the old model

    :x_train: numpy array (input of shape (x, y, z))
    :y_train: numpy array (prediction labels of shape (x, z))
    :model_name: String

    :return: None
    """
    # architecture
    lstm_input = Input(shape=(x_train.shape[1], 5), name='input_lstm')
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
    hist = model.fit(x=x_train, y=y_train, batch_size=32, epochs=50)

    if model_name is None:
        model.save('model_default.h5')
    else:
        model.save(model_name)

