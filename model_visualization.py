#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, Multiply
from keras.layers import Bidirectional, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

def SEBlock(inputs, reduction=16, if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // reduction, use_bias=False, activation=tf.keras.activations.relu, trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid, trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs, x])

inputs = Input(name='inputs',shape=[360,1], dtype='float64')
#-----------------------CNN1d----------------------------#
'''
layer1 = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu)(
    inputs)
BachNorm = tf.keras.layers.BatchNormalization()(layer1)
MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(BachNorm)
layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(
    MaxPooling1)
BachNorm = tf.keras.layers.BatchNormalization()(layer2)
MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(BachNorm)
layer3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(
    MaxPooling2)
layer4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(layer3)
MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(layer4)
layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(
    MaxPooling3)
layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(layer5)
flat = tf.keras.layers.Flatten()(layer6)
x = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flat)
x = tf.keras.layers.Dropout(rate=0.1)(x)
'''
#-------------------------------------------------------------------#

#-------------------------SeqCNN------------------------------------#
'''
x=tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same',activation=tf.nn.relu)(inputs)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
x=tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
x=tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
x=tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(x)
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
x=tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(x)
x=tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(x)
x=tf.keras.layers.Dropout(rate=0.1)(x)
x=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
'''
#--------------------------------------------------------------------#

#-------------------------CNN+LSTM-----------------------------------#
'''
x = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu)(
    inputs)
# x=tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same',activation=tf.nn.relu)(input_ecg)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
# tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
# tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
# tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
x = tf.keras.layers.LSTM(10)(x)
x = tf.keras.layers.Flatten()(x)
# tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
x = tf.keras.layers.Dropout(rate=0.1)(x)
x = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
'''
#--------------------------------------------------------------------#

#--------------------SENet+LSTM-------------------------------#
'''
x=tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same',activation=tf.nn.relu)(inputs)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
x=tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
x=SEBlock(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
x=tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
x = SEBlock(x)
# tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
# tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
# tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
x=tf.keras.layers.LSTM(10)(x)
x=tf.keras.layers.Flatten()(x)
# tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
x=tf.keras.layers.Dropout(rate=0.1)(x)
x=tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(x)
x=tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(x)
x=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
'''
#-------------------------------------------------------------#

#---------------------Seq+SE+GRU-----------------------------#
x = tf.keras.layers.Conv1D(filters=256, kernel_size=50, strides=3, padding='same',activation=tf.nn.relu)(inputs)
x = SEBlock(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
x = tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu)(x)
x = SEBlock(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(x)
x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
x = tf.keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(x)
x = tf.keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(x)
x = SEBlock(x)
x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
x = tf.keras.layers.GRU(units=70)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=45, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(rate=0.3)(x)
x= tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
#-------------------------------------------------------------#

output= tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
model = Model(inputs=[inputs], outputs=output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
plot_model(model,'Seq_SE_GRU_model.png',show_shapes=True)
