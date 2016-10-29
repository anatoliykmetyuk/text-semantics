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

  m.compile(loss='categorical_crossentropy', optimizer='adam')

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

def strings_identity(layer, sentences=['ABGLDJTMBLDTYGA'], nb_epoch=1000, **kwargs):
  length = len(max(sentences, key=len))
  print("Sentences size:", len(sentences))
  print("Max length:", length)

  # Load data
  converter = CharEnc(charMapList(sentences), length)
  X         = converter.textToOneHot(sentences)

  # Define the model
  input_shape = (length, converter.embeddingLen)
  output_dim  = converter.embeddingLen
  m = one_layer_categorical(layer, **dict(kwargs, input_shape=input_shape, output_dim=output_dim))

  # Train the model
  m.fit(X, X, nb_epoch=nb_epoch)
  
  # Metrics
  y = m.predict(X)

  def flatten(x):
    x_flat = converter.oneHotToId(x)
    return x_flat.reshape(x_flat.size)
    
  X_flat = flatten(X)
  y_flat = flatten(y)

  # Sample
  print(converter.charMap)
  print(converter.oneHotToText(X[:1]), converter.oneHotToId(X[:1]))
  print(converter.oneHotToText(y[:1]), converter.oneHotToId(y[:1]))

  # Accuracy
  print("Accuracy", accuracy_score(X_flat, y_flat))
