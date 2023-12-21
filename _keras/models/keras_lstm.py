

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Bidirectional, Dropout, RepeatVector, Input, Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import TimeDistributed
tf.random.set_seed(2012)

# github https://github.com/JeCase/LoadElectricity_Forecasting_CNN-BiLSTM-Attention/blob/main/Forecasting_CNN-BiLSTM.ipynb


# https://github.com/JeCase/LoadElectricity_Forecasting_CNN-BiLSTM-Attention/blob/main/Forecasting_CNN-BiLSTM%2BAttention.ipynb
def CNN_LSTM_att(structure, args):
    input_shape = (args.seq_len, args.feature_no)
    model = Sequential([
        Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=input_shape),
        Conv1D(filters=8, kernel_size=1, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Attention(),
        Dropout(structure['dropout']),
        Bidirectional(LSTM(structure['n_hidden_units'])),
        Dense(structure['n_hidden_units']/2, activation='relu'),
        Dense(1)
    ])
    return model


def keras_cnn_lstm(structure , args):
        input_shape = (args.seq_len, args.feature_no)
        model = Sequential()
        model.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=input_shape))
        model.add(Dropout(structure['dropout']))
        model.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=input_shape))
        model.add(Bidirectional(LSTM(structure["n_hidden_units"], return_sequences=True)))
        model.add(Dropout(structure['dropout']))
        model.add(Bidirectional(LSTM(structure["n_hidden_units"])))
        model.add(Dense(structure["h2"], activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model

def encoder_decoder_LSTM(structure , args):
    n_timesteps, n_features, n_outputs = args.seq_len, args.feature_no, args.pred_len
    model = Sequential()
    model.add(LSTM(structure["n_hidden_units"], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(structure["dropout"]))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
    model.add(TimeDistributed(tf.keras.layers.Dense(1)))
    return model


def encoder_decoder_GRU(structure , args):
    n_timesteps, n_features, n_outputs = args.seq_len, args.feature_no, args.pred_len
    model = Sequential()
    model.add(GRU(structure["n_hidden_units"], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(structure["dropout"]))
    model.add(RepeatVector(n_outputs))
    model.add(GRU(structure["h2"], activation='relu', return_sequences=True))
    model.add(TimeDistributed(tf.keras.layers.Dense(structure["h2"]/2, activation='relu')))
    model.add(TimeDistributed(tf.keras.layers.Dense(1)))
    return model


class Attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        logits = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(logits, axis=1)
        weighted_inputs = inputs * attention_weights
        if self.return_sequences:
            return weighted_inputs
        else:
            return tf.reduce_sum(weighted_inputs, axis=1)



