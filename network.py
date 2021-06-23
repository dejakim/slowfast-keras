'''
dert.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

# Tensorflow dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

W = 112
C = 16
L2_RATE = 0.001
DO_RATE = 0.1


def bottleneck(inputs, filters, kernel, strides, r=False):
  depth = filters // 4

  x = tf.keras.layers.Conv3D(depth, kernel, strides=strides, padding='same')(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv3D(depth,(1,3,3), strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv3D(filters, 1, strides=1, padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(L2_RATE))(x)
  x = tf.keras.layers.BatchNormalization()(x)

  if r:
    x = tf.keras.layers.Add()([x, inputs])
  
  return tf.keras.layers.Activation('relu')(x)

def residual_block(inputs, filters, kernel, strides, n):
  x = bottleneck(inputs, filters, kernel, strides, False)
  for _ in range(n-1):
    x = bottleneck(x, filters, kernel, 1, True)
  return x

def conv_block(inputs, filters, kernel, strides, padding='same'):
  x = tf.keras.layers.Conv3D(
    filters,
    kernel,
    strides=strides,
    padding=padding)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  return tf.keras.layers.Activation('relu')(x)


def create_model(input_shape=(C,W,W,1), k=27):
  #tf.keras.mixed_precision.set_global_policy('mixed_float16')

  inputs = tf.keras.layers.Input(shape=input_shape)

  # Slow path
  s = tf.keras.layers.Conv3D(1, 1, strides=(8,1,1))(inputs)
  s = conv_block(s, 64, (1,7,7), (1,2,2))
  s = tf.keras.layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')(s)
  s = residual_block(s, 256, (1,1,1), (1,1,1), 3)
  s = residual_block(s, 512, (1,1,1), (1,2,2), 4)
  s = residual_block(s,1024, (3,1,1), (1,2,2), 6)
  # s = residual_block(s,2048, (3,1,1), (1,2,2), 3)
  # s = tf.keras.layers.Dropout(DO_RATE)(s)
  s = tf.keras.layers.GlobalAveragePooling3D()(s)

  # Fast path
  f = inputs
  f = conv_block(f, 8, (5,7,7), (1,2,2))
  f = tf.keras.layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')(f)
  f = residual_block(f, 32, (3,1,1), (1,1,1), 3)
  f = residual_block(f, 64, (3,1,1), (1,2,2), 4)
  f = residual_block(f,128, (3,1,1), (1,2,2), 6)
  # f = residual_block(f,256, (3,1,1), (1,2,2), 3)
  # f = tf.keras.layers.Dropout(DO_RATE)(f)
  f = tf.keras.layers.GlobalAveragePooling3D()(f)

  y = tf.keras.layers.Concatenate()([s, f])
  outputs = tf.keras.layers.Dense(k, activation='softmax')(y)
  
  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
  model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

  return model

if __name__ == '__main__':
  # Create model
  model = create_model()
  model.summary()
