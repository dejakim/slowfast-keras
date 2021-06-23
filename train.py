'''
train.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python train.py
'''
from __future__ import print_function

import os
import numpy as np
import cv2
import pickle
import gzip
import random
from os.path import join
import matplotlib.pyplot as plt

import tensorflow as tf

# from prepare import draw
from network import create_model, C
from datagenerator import DataGenerator

# Tensorflow dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

def load_data(bin_path):
  if os.path.isfile(bin_path):
    with gzip.open(bin_path, 'rb') as f:
      data = pickle.load(f)
      return data
  exit("Could not load data at: {}".format(bin_path))

if __name__ == '__main__':
  batch_size = 16
  validation_split = .2
  use_generator = True

  check_path = 'model/weights.h5'
  save_path = 'model/network.h5'

  ###################
  # Create model
  print('-'*30)
  print('Create model')
  model = create_model()
  # model.summary()
  # exit('quit')

  #########################
  # Load data for trainingint('-'*30)
  print('Load train data')
  data = load_data('./data/jester.pickle')
  x_train, y_train = data["x_train"], data["y_train"]
  x_valid, y_valid = data["x_valid"], data["y_valid"]

  # size of dataset
  print(x_train.shape, x_valid.shape)

  ###################
  # Start Training
  print('-'*30)
  print('Training start')
  # callbacks
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    check_path, monitor='val_loss', save_best_only=True)
  earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, verbose=0, mode='auto')
  # loading weights of the last check point
  if os.path.isfile(check_path):
    model.load_weights(check_path)

  # training schedule tuple of learning rate and iterations
  schedule = [(2e-3, 300), (1e-4, 100), (1e-5, 150)]
  lr, epochs = (1e-4, 20)#schedule[0] 2e-3, 20 -> 2e-3/5, 15 -> 2e-3/25, 20

  tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

  if use_generator:
    # data generators
    training_generator = DataGenerator(x_train, y_train, batch_size=batch_size, num_sequences=C)
    validation_generator = DataGenerator(x_valid, y_valid, batch_size=batch_size, num_sequences=C)
    hist = model.fit(
      training_generator,
      None,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      #use_multiprocessing=True,
      #workers=2,
      callbacks=[checkpoint, earlystop]
    )
  else:
    hist = model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=validation_split,
      callbacks=[checkpoint, earlystop]
    )
  
  if not os.path.exists('model'):
    os.makedirs('model')
  model.save(save_path)
  print('Trainig finished.')

  # Loss History
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.plot(hist.history['sparse_categorical_accuracy'])
  plt.plot(hist.history['val_sparse_categorical_accuracy'])
  plt.title('model loss')
  plt.ylabel('rate')
  plt.xlabel('epoch')
  plt.legend(['train', 'val', 'train_acc', 'val_acc'], loc='upper left')
  plt.show()
