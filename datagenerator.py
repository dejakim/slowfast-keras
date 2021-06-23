'''
datagenerator.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np
import cv2
import pickle
import gzip

import tensorflow as tf
import matplotlib.pyplot as plt

def draw(sample):
  # H,W,N = sample.shape[:3]
  N,H,W = sample.shape[:3]
  fig = plt.figure()
  cols, rows = 5, (N+4)//5
  for i in range(N):
    ax = fig.add_subplot(rows, cols, i+1)
    # ax.imshow(sample[:,:,i].reshape(H,W))
    ax.imshow(sample[i].reshape(H,W))
    ax.set_xticks([]), ax.set_yticks([])
  plt.show()

def rotate(images, angle):
  h, w = images[0].shape[:2]
  c = np.sqrt(h * h + w * w)
  s = c / h * np.cos( np.arctan(w/h) - abs(np.deg2rad(angle)) )
  M = cv2.getRotationMatrix2D((w/2, h/2), angle, s)
  for i, img in enumerate(images):
    img = cv2.warpAffine(img, M, (w,h)).reshape((h,w,1))
    img = img / np.float32(128.) - np.float32(1.) #(img - np.mean(img)) / 255.
    images[i] = img
  # return np.concatenate(images, axis=2)
  return np.stack(images, axis=0)

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, labels, batch_size=16, shuffle=True, num_sequences=32):
    '''Initialization'''
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    # custom params
    self.num_sequences = num_sequences * 2
    # initial indexes
    self.on_epoch_end()

  def __len__(self):
    return len(self.images) // self.batch_size

  def __getitem__(self, index):
    # Generate indexes of the batch
    start = index * self.batch_size
    indexes = self.indexes[ start : start + self.batch_size ]

    # Generate data
    return self.__data_generation(indexes)

  def on_epoch_end(self):
    '''Updates indexes after each epoch'''
    self.indexes = np.arange(len(self.images))
    if self.shuffle == True:
      # np.random.seed(self.random_state)
      np.random.shuffle(self.indexes)

  def __data_generation(self, indexes):
    angles = (np.random.rand(len(indexes)) - .5) * 10. # -10 ~ +10deg
    shifts = np.random.randint(5, size=len(indexes))
    x, y = [], []
    for i, angle, shift in zip(indexes, angles, shifts):
      # load images
      samples = []
      # P = (len(self.images[i]) - self.num_sequences)//2 - shift
      P = min(shift, len(self.images[i]) - self.num_sequences-1)
      Q = P + self.num_sequences
      # for img_path in sorted(random.sample(list(self.images[i,P:Q]), self.num_sequences)):
      for j, img_path in enumerate(list(self.images[i, P:Q])):
        if (j & 1) == 1:
          continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
          exit("Could not load image file: ", img_path)
        samples.append(img)
      packed = rotate(samples, angle)
      # print("shape = ", packed.shape)
      x.append(packed)
      y.append(self.labels[i])
      # print("img = {}, label = {}".format(self.images[i,0], y[-1]))
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)

if __name__ == "__main__":
  bin_path = './data/jester.pickle'

  if not os.path.isfile(bin_path):
    exit("Could not load data at: " + bin_path)
  
  print("load files")
  with gzip.open(bin_path, 'rb') as f:
    data = pickle.load(f)
    x_train, y_train = data["x_train"], data["y_train"]
    x_valid, y_valid = data["x_valid"], data["y_valid"]
    # x_test , y_test  = data["x_test"], data["y_test"]
  
  print("generate image data")
  datagen = DataGenerator(x_valid, y_valid)
  for i, (x,y) in enumerate(datagen):
    if i > 0:
      break
    for x_i, y_i in zip(x,y):
      draw(x_i)
