import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Model:

    def __init__(self):
        self.lstm_model = None

    def build_model(self, vocab_size=3155):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.lstm_model = Sequential()

        self.lstm_model.add(Embedding(vocab_size, 256, input_length=15))

        self.lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))

        self.lstm_model.add(LSTM(128))
        self.lstm_model.add(Dense(1000, activation='relu'))
        self.lstm_model.add(Dense(vocab_size, activation='softmax'))

        self.lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.lstm_model.summary()

    def fit_model(self, training_data, label_data):
        self.lstm_model.fit(training_data, label_data, epochs=100)
        self.lstm_model.save('lstm_model.h5')