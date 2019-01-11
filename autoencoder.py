# -*- coding: utf-8 -*-

import numpy
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.core import Dense
from keras.constraints import non_neg

class Autoencoder:
    def __init__(self, setting):
        self.setting = setting
        self.with_batch = False
        self.simple()

    def simple(self):
        self.main_input = Input(shape=self.setting.input_shape[1:])
        self.encoded = Dense(512, activation=self.setting.hidden_activation)(self.main_input)
        self.encoded = Dense(256, activation=self.setting.hidden_activation)(self.encoded)
        self.encoded = Dense(128, activation=self.setting.hidden_activation)(self.encoded)
        self.decoded = Dense(256, activation=self.setting.hidden_activation)(self.encoded)
        self.decoded = Dense(512, activation=self.setting.hidden_activation)(self.decoded)
        self.decoded = Dense(self.setting.input_neurons, activation=self.setting.output_activation)(self.decoded)
        self.autoencoder = Model(inputs=self.main_input, outputs=self.decoded)
        self.autoencoder.compile(optimizer=self.setting.optimizer, loss=self.setting.loss, metrics=["accuracy"])

    def reshape(self, data, shape):
        return numpy.array(data).reshape(shape)

    def input_reshape(self, data):
      if self.with_batch:
        return self.reshape(data, self.setting.batch_input_shape)
      return self.reshape(data, self.setting.input_shape)

    def train(self, x_train, x_test):
        x_train = self.input_reshape(x_train)
        x_test = self.input_reshape(x_test)
        validation_data = (x_test, x_test)
        self.autoencoder.fit(x_train, x_train, nb_epoch=self.setting.nb_epoch, batch_size=self.setting.batch_size, validation_data=validation_data)
        self.autoencoder.save_weights('simulator/weights/autoencoder.h5')

    def predict(self, x_train):
        self.autoencoder.load_weights('simulator/weights/autoencoder.h5')
        encoder = Model(input=self.main_input, output=self.encoded)
        x_train = self.input_reshape(x_train)
        return encoder.predict(x_train)

    def evaluate(self, x_test):
        self.autoencoder.load_weights('simulator/weights/autoencoder.h5')
        x_test = self.input_reshape(x_test)
        return self.autoencoder.evaluate(x_test, x_test)
