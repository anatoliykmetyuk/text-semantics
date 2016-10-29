from keras.layers import *
from keras.models import Model, Sequential

from sklearn.metrics import accuracy_score

import numpy as np
from dataencoders import CharEnc, charMapList

np.random.seed(1337)

def model(layers):
  m = Sequential()

  for l in layers:
    m.add(l)

  m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  print(m.summary())
  return m

def one_layer_categorical(layer, **kwargs):
  return model([layer(**dict(kwargs, activation='softmax'))])

def one_layer_masking(layer, **kwargs):
  input_shape = kwargs.pop('input_shape')
  return model([
    Masking(mask_value=0, input_shape=input_shape)
  , layer(**dict(kwargs, activation='softmax'))
  ])

def lstm(**kwargs): return LSTM     (return_sequences=True, **kwargs)
def rnn (**kwargs): return SimpleRNN(return_sequences=True, **kwargs)
def dense_first(**kwargs):
  input_shape = kwargs.pop('input_shape')
  return TimeDistributed(Dense(**kwargs), input_shape=input_shape)
def dense(**kwargs): return TimeDistributed(Dense(**kwargs))

def strings_identity(layer, sentences, nb_epoch=1000, test_and_valid_size=[0.15, 0.15], **kwargs):
  length = len(max(sentences, key=len))
  print("Sentences size:", len(sentences))
  print("Max length:", length)

  # Load data
  converter = CharEnc(charMapList(sentences), length)
  all_data  = converter.textToOneHot(sentences)
  np.random.shuffle(all_data)

  valid_size = int(test_and_valid_size[0] * len(all_data))
  test_size  = int(test_and_valid_size[1] * len(all_data))

  X_test  = all_data[:test_size]
  X_valid = all_data[test_size : test_size + valid_size]
  X_train = all_data[test_size + valid_size:]

  print("Test size:", test_size)
  print("Validation size:", valid_size)
  print("Train size", len(X_train))

  # Define the model
  input_shape = (length, converter.embeddingLen)
  output_dim  = converter.embeddingLen
  m = one_layer_categorical(layer, **dict(kwargs, input_shape=input_shape, output_dim=output_dim))

  # Train the model
  m.fit(X_train, X_train
  , nb_epoch=nb_epoch
  , validation_data=(X_valid, X_valid)
  , batch_size=64)

  # Metrics
  y = m.predict(X_test)

  def flatten(x):
    x_flat = converter.oneHotToId(x)
    return x_flat.reshape(x_flat.size)
    
  X_flat = flatten(X_test)
  y_flat = flatten(y)

  # Sample
  print(converter.charMap)
  print(converter.oneHotToText(X_test[:10]))
  print(converter.oneHotToText(y[:10]))

  # Accuracy
  print("Test Accuracy", accuracy_score(X_flat, y_flat))
