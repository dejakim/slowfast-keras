'''
app.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import numpy as np
import cv2
import tensorflow as tf
import threading
import time

LABELS = [
  'Doing other things',
  'Drumming Fingers',
  'No gesture',
  'Pulling Hand In',
  'Pulling Two Fingers In',
  'Pushing Hand Away',
  'Pushing Two Fingers Away',
  'Rolling Hand Backward',
  'Rolling Hand Forward',
  'Shaking Hand',
  'Sliding Two Fingers Down',
  'Sliding Two Fingers Left',
  'Sliding Two Fingers Right',
  'Sliding Two Fingers Up',
  'Stop Sign',
  'Swiping Down',
  'Swiping Left',
  'Swiping Right',
  'Swiping Up',
  'Thumb Down',
  'Thumb Up',
  'Turning Hand Clockwise',
  'Turning Hand Counterclockwise',
  'Zooming In With Full Hand',
  'Zooming In With Two Fingers',
  'Zooming Out With Full Hand',
  'Zooming Out With Two Fingers'
]

class Detector:
  def __init__(self):
    # load saved model
    model = tf.keras.models.load_model(
      './model/weights.h5')
    self._model = model
    self._shape = model.input.get_shape()[1:]
    print('model loaded: input shape=', self._shape)
  
  # Get input shape
  # None, num of seq, width, height, channel
  def get_shape(self):
    return self._shape
  
  def inference(self, images):
    N,W,H,C = self._shape
    if len(images) < N:
      return (-1, 0.0)
    # Stack image
    src = np.reshape(np.stack(images[-N:], axis=0), (-1,N,W,H,C))
    # Inference
    infer = self._model.predict(src)[0]
    index = np.argmax(infer)
    return (index, infer[index])

if __name__ == '__main__':
  # Load model
  detector = Detector()
  # camera open
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  N,W,H = detector.get_shape()[:3]
  images = []
  label = [""]
  running = True
  thres = 0.8

  def infer():
    while running:
      if len(images) >= N:
        index, score = detector.inference(images)
        if index >= 0 and index < len(LABELS) and score > thres:
          label[0] = LABELS[index]
          time.sleep(0.05)
  
  # inference thread
  th = threading.Thread(target=infer)
  th.start()

  # main loop
  count = 0
  while True:
    ret, frame = capture.read()
    # resize & normalize image
    if count == 0:
      sample = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (W,H))
      sample = sample / np.float32(128.) - np.float32(1.)
      images = images[1-N:] + [sample]
    count = (count + 1) % 3
    # Inference
    img = cv2.putText(frame, "Gesture: " + label[0], (0,30),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow("VideoFrame", img)
    if cv2.waitKey(1) > 0:
      running = False
      break
  
  # wait until inference thread finish its task
  th.join()

  capture.release()
  cv2.destroyAllWindows()