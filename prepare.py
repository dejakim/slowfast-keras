'''
prepare.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import pickle
import gzip
import json
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

def draw(samples):
  fig = plt.figure()
  rows, cols = 4, 4
  for i, img in enumerate(samples):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(img)
    ax.set_xticks([]), ax.set_yticks([])
  plt.show()

def load_jester(csv_path, dst_path):
  print("loading jester dataset :", csv_path)
  dir_path = os.path.splitext(csv_path)[0]
  # load data
  if not os.path.isfile(csv_path):
    exit('could not open dataset file: {}'.format(csv_path))
  if not os.path.isdir(dir_path):
    exit('could not find image directory: '.format(dir_path))

  images, labels = [],[]
  
  lines = csv.reader(open(csv_path, 'r', encoding='utf-8'))
  exts = {".jpg", ".jpeg", ".png", ".gif"}
  for line in tqdm(lines):
    if not line[0].isnumeric():
      continue
    vid,_,frames,label,_,_ = line[:6]
    base_dir = os.path.join(dir_path, vid)
    dest_dir = os.path.join(dst_path, vid)
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)
    file_list = sorted(os.listdir(base_dir))
    img_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in exts)]
    saved = []
    for name in img_files:
      src_path = os.path.join(base_dir, name)
      trg_path = os.path.join(dest_dir, name)
      img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        print('could not read file: ', src_path)
        break
      img = cv2.resize(img, (112,112))
      cv2.imwrite(trg_path, img)
      saved.append(trg_path)
    
    if len(saved) < 16:
      print('too many missed files... skip this video')
      continue

    images.append(saved)
    labels.append(label)
  return np.array(images), np.array(labels)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    exit("usage: {} <jester dataset path>")
  
  src_dir=sys.argv[1]
  bin_path = './data/jester.pickle'

  print('-'*30)
  print('load meta data')
  
  x_train, y_train = load_jester(os.path.join(src_dir, 'Train.csv'), './data/Train')
  print(x_train.shape, y_train.shape)
  
  x_valid, y_valid = load_jester(os.path.join(src_dir, 'Validation.csv'), './data/Validation')
  print(x_valid.shape, y_valid.shape)
  
  x_test, y_test = load_jester(os.path.join(src_dir, 'Test.csv'), './data/Test')
  print(x_test.shape, y_test.shape)
  
  print('-'*30)
  print('save to file')
  # save to file
  with gzip.open(bin_path, 'wb') as f:
    pickle.dump({
      "x_train": x_train, "y_train": y_train,
      "x_valid": x_valid, "y_valid": y_valid,
      "x_test" : x_test , "y_test" : y_test
    }, f)
    print('file saved')
  
  print('done')
