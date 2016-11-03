from keras.layers import *
from keras.models import Model, Sequential

import numpy as np

from util import dataset
from dataencoders import CharEnc, charMapList

import prims

nb_epoch = 100

def _run():
  # Load data
  raw = dataset.loadLines(dataset.BareSentencesPath)
  X, y, converter = prims.one_hot_txt(raw)
  y = prims.shift(y, 1)
  X_train, y_train, X_valid, y_valid, X_test, y_test = prims.split_data(X, y)

  # Compose a model
  inp = Input(shape=X_train.shape[1:])
  l1  = TimeDistributed(Dense(128, activation='relu'))(inp)
  l2  = GRU(128, activation='relu', return_sequences=True)(l1)
  l3  = TimeDistributed(Dense(converter.embeddingLen, activation='softmax'))(l2)
  m   = Model(input=inp, output=l3)

  m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Train the model
  m.fit(X_train, y_train
  , nb_epoch=nb_epoch
  , validation_data=(X_valid, y_valid)
  , batch_size=64)

  # Report the metrics
  y_test_pred     = m.predict(X_test)
  y_test_pred_raw = prims.one_hot_to_txt(y_test_pred, converter)
  y_test_raw      = prims.one_hot_to_txt(y_test, converter)

  print(y_test_raw[:10])
  print(y_test_pred_raw[:10])
  print("Test Accuracy", accuracy_score(prims.flatten(y_test, converter), prims.flatten(y_test_pred, converter)))

_run()
