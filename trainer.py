# -*- coding: utf-8 -*-
import os
import numpy
import tensorflow
import random
import time
import subprocess
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Input, Conv1D, Conv2D, Flatten, Dropout, GlobalMaxPooling1D, MaxPooling1D, MaxPooling2D, Reshape, BatchNormalization, concatenate, Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.layers.merge import add
from keras.layers.pooling import AveragePooling1D, AveragePooling2D
from sklearn.metrics import precision_score
import keras.backend as K

class TrainerSetting:
    def __init__(self):
        self.length_of_sequences = 1
        self.input_neurons = 1
        self.output_neurons = 1 
        self.hidden_neurons = 300
        self.encode_neurons = 25
        self.batch_size = 1
        self.nb_epoch = 500
        self.depth = 0
        self.kernel_size = 3
        self.filter_size = 32
        self.dropout = 0.02
        self.input_shape = (-1, self.input_neurons)
        self.batch_input_shape = (-1, self.input_neurons, 1)
        self.output_shape = (-1)
        self.window_size = 5
        self.data_length = 50
        self.split = 0.8
        self.class_weight = None
        self.sample_weight = None
        self.hidden_activation = "sigmoid"
        self.output_activation = "sigmoid"
        self.metrics = ["accuracy"]
        self.loss = "binary_crossentropy"
        self.optimizer = "adam"
        self.kernel_initializer = "glorot_uniform"
        self.bias_initializer = "zeros"
        self.regularizer_l1 = 0.01
        self.regularizer_l2 = 0.01
        self.loss_weights = None
        self.with_batch = False
        self.with_sequences = False
        self.with_multi_input = False
        self.weights_filename = "stocks_simulation.hdf5"
        self.choice = []
        self.categorical = True

class ModelCreator:

  @staticmethod
  def resnet(setting):
    return ResNet50(input_shape=setting.batch_input_shape[1:], weights=None, classes=setting.output_neurons)

  @staticmethod
  def cnn2d(setting):
    model = Sequential()
    model.add(Conv2D(setting.filter_size, setting.kernel_size, padding="same", activation=setting.hidden_activation, input_shape=setting.batch_input_shape[1:],
        kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer,
        kernel_regularizer=regularizers.l1_l2(setting.regularizer_l1, setting.regularizer_l2), bias_regularizer=regularizers.l1_l2(setting.regularizer_l1, setting.regularizer_l2)))
    model.add(MaxPooling2D(padding="same"))
    model.add(Dropout(setting.dropout))
    model.add(Flatten())
    model.add(Dense(setting.output_neurons, activation=setting.output_activation))
    return model

  @staticmethod
  def cnn(setting):
    model = Sequential()
    model.add(Conv1D(setting.filter_size, setting.kernel_size, padding="same", activation=setting.hidden_activation, input_shape=setting.batch_input_shape[1:],
        kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer))
    model.add(MaxPooling1D(pool_length=setting.kernel_size, padding="same"))
    model.add(Dropout(setting.dropout))

    for i in range(setting.depth):
        model.add(Conv1D(setting.filter_size, setting.kernel_size, padding="same", activation=setting.hidden_activation, 
            kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer))
        model.add(MaxPooling1D(padding="same"))
        model.add(Dropout(setting.dropout))

    model.add(Flatten())
    model.add(Dense(setting.output_neurons, activation=setting.output_activation))
    return model

  @staticmethod
  def ndcnn_lstm(setting):
    inputs, cnns = [], []
    for i in range(setting.depth):
        cnn_input = Input(shape=setting.batch_input_shape[2:]) # (length_sequence, data_num, pattern_size, 1[channels])
        x = Conv1D(setting.filter_size, setting.kernel_size, padding='same', activation=setting.hidden_activation)(cnn_input)
        x = MaxPooling1D(padding="same")(x)
        x = Dropout(setting.dropout)(x)
        x = Flatten()(x)
        inputs.append(cnn_input)
        cnns.append(x)
    merged = concatenate(cnns)
    reshaped = Reshape((setting.depth, -1))(merged)
    lstm = LSTM(setting.hidden_neurons, activation=setting.hidden_activation, return_sequences=True)(reshaped)
    pool = GlobalMaxPooling1D()(lstm)
    outputs = Dense(setting.output_neurons, activation=setting.output_activation)(pool)
#    outputs = Dense(setting.output_neurons, activation=setting.output_activation)(merged)
    return Model(inputs=inputs, outputs=outputs)

  @staticmethod
  def cnn_lstm(setting):
    model = Sequential()
    model.add(Embedding(2000, 128, input_length=setting.data_length))
    model.add(Dropout(setting.dropout))
    model.add(Conv1D(setting.filter_size, setting.kernel_size, padding='same', activation=setting.hidden_activation, input_shape=setting.batch_input_shape[1:],
        kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer))
    model.add(MaxPooling1D(padding="same"))
    model.add(LSTM(setting.hidden_neurons, return_sequences=False))
    model.add(Dense(setting.output_neurons, activation=setting.output_activation))
    return model

  @staticmethod
  def lstm(setting):
    model = Sequential()
    model.add(LSTM(setting.hidden_neurons, return_sequences=True if setting.depth > 0 else False,
              input_shape=setting.batch_input_shape[1:]))
    model.add(Dropout(setting.dropout))
    for i in range(setting.depth):
        model.add(LSTM(setting.hidden_neurons, return_sequences=True if i < setting.depth - 1 else False))
        model.add(Dropout(setting.dropout))
    model.add(Dense(setting.output_neurons, activation=setting.output_activation))
    return model

  @staticmethod
  def multimodal(setting):
    inputs = []
    hiddens = []
    for _ in range(len(setting.choice)):
        main_input = Input(shape=setting.batch_input_shape[1:])
        x = LSTM(setting.hidden_neurons, activation=setting.hidden_activation)(main_input)
        inputs.append(main_input)
        hiddens.append(x)
    merged = add(hiddens)
    output = Dense(setting.output_neurons, activation=setting.output_activation)(merged)
    return Model(inputs=inputs, outputs=output)

  @staticmethod
  def dnn(setting):
    activation = setting.hidden_activation
    model = Sequential()
    model.add(Dense(setting.hidden_neurons, input_shape=setting.input_shape[1:],
        kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer,
        kernel_regularizer=regularizers.l1_l2(setting.regularizer_l1, setting.regularizer_l2), bias_regularizer=regularizers.l1_l2(setting.regularizer_l1, setting.regularizer_l2)))
    model.add(Activation(activation))
    model.add(Dropout(setting.dropout))

    for i in range(setting.depth):
      model.add(Dense(setting.hidden_neurons, 
        kernel_initializer=setting.kernel_initializer, bias_initializer=setting.bias_initializer))
      model.add(Activation(activation))
      model.add(Dropout(setting.dropout))

    model.add(Dense(setting.output_neurons))
    model.add(Activation(setting.output_activation))
    return model

  @staticmethod
  def stacked(setting):
    # https://keras.io/ja/getting-started/functional-api-guide/#_1
    main_input = Input(shape=setting.input_shape[1:])
    x = Dense(setting.hidden_neurons, activation=setting.hidden_activation)(main_input)
    x = Dropout(setting.dropout)(x)
    sub_output = Dense(setting.output_neurons, activation=setting.output_activation)(x)

    inputs = [main_input]
    outputs = [sub_output]
    for i in range(setting.depth):
        sub_input = Input(shape=setting.input_shape[1:])
        x = concatenate([sub_output, sub_input])
        x = Dense(setting.hidden_neurons, activation=setting.hidden_activation)(x)
        x = Dropout(setting.dropout)(x)
        sub_output = Dense(setting.output_neurons, activation=setting.output_activation)(x)
        inputs.append(sub_input)
        outputs.append(sub_output)

    return Model(inputs=inputs, outputs=outputs)


class Trainer:
  def __init__(self, setting, callback):
    self.setting = setting
    self.set_random_seed(int(time.time()))
    self.model = self.create_model(callback)

  # https://qiita.com/TokyoMickey/items/63c4053740ab1f3f28a2
  # GPUを使うと毎回結果が変わるらしいので安定させるためにseedを合わせる
  def set_random_seed(self, seed):
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow.set_random_seed(seed)

  # モデル作成
  def create_model(self, callback):
    model = callback(self.setting)
    model.compile(loss=self.setting.loss, optimizer=self.setting.optimizer, metrics=self.setting.metrics, loss_weights=self.setting.loss_weights)
    model.summary()
    return model

  def reshape(self, data, shape):
    return numpy.array(data).reshape(shape)

  def input_reshape(self, data):
    if self.setting.with_batch:
      print("use with batch")
      batch_input = self.reshape(data, self.setting.batch_input_shape)
      if self.setting.with_multi_input:
        return [x for x in batch_input]
      else:
        return batch_input
    return self.reshape(data, self.setting.input_shape)

  def output_reshape(self, data):
    output = self.reshape(data, self.setting.output_shape)
    if self.setting.categorical:
        return to_categorical(output)
    else:
        return output

  # 学習
  def train(self, x_train, y_train, x_test, y_test, early_stopping=False, tensorboad=False):
    x_train = self.input_reshape(x_train)
    y_train = self.output_reshape(y_train)
    x_test = self.input_reshape(x_test)
    y_test = self.output_reshape(y_test)

    print("x_train: %s, y_train: %s, x_test: %s, y_test: %s" % (len(x_train), len(y_train), len(x_test), len(y_test)))

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(patience=3))
    if tensorboad:
        subprocess.call(["rm", "-rf", "/tmp/simulation"])
        callbacks.append(TensorBoard(log_dir="/tmp/simulation", histogram_freq=1))
    shuffle = True
    validation_data = (x_test, y_test)
    self.model.fit(x_train, y_train, batch_size=self.setting.batch_size, nb_epoch=self.setting.nb_epoch,
        validation_data=validation_data,
        callbacks=callbacks,
        shuffle=shuffle,
        class_weight=self.setting.class_weight,
        sample_weight=self.setting.sample_weight)
    self.save_model()

  def predict(self, x):
    x = self.input_reshape(x)
    return self.model.predict(x, batch_size=self.setting.batch_size, verbose=1)

  def evaluate(self, x, y):
    x = self.input_reshape(x)
    y = self.output_reshape(y)
    return self.model.evaluate(x, y)

  def save_model(self):
    # モデルとパラメータを出力
    json_string = self.model.to_json()
    open(self.model_path(), 'w').write(json_string)
    self.model.save_weights(self.weights_path())

  def load_model(self):
    json_string = open(self.model_path()).read()
    model = model_from_json(json_string)
    model.load_weights(self.weights_path())
    return model

  def load_weights(self):
    self.model.load_weights(self.weights_path())

  def model_path(self):
    output_dir = "simulator/model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, "stocks_simulation_model.json")

  def weights_path(self):
    print(self.setting.weights_filename)
    output_dir = "simulator/weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, self.setting.weights_filename)
