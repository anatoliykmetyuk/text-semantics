from keras.layers import *
from keras.models import Model, Sequential

import numpy as np

from util import dataset
from dataencoders import CharEnc, charMapList

samples_size  = 1
length        = 15
nb_epoch      = 400

embedding_dim = 10
hidden_dim    = embedding_dim

np.random.seed(1337)

def model(in_shape, out_shape, **kwargs):
  m = Sequential()

  # Performs even worse with embeddings
  # m.add(TimeDistributed(Dense(embedding_dim), input_shape=in_shape))
  # m.add(Activation('tanh'))

  m.add(LSTM(
      output_dim = in_shape[-1]
    , input_dim  = in_shape[-1]
    , input_length = in_shape[0]
    , return_sequences = True
    , **kwargs))

  # m.add(TimeDistributed(Dense(out_shape[-1])))
  # m.add(Activation('tanh'))

  m.compile(loss='categorical_crossentropy', optimizer='adam')

  print(m.summary())
  return m

def _test():
  # Load data
  sentences = ['ABAB']#dataset.loadLines(dataset.BareSentencesPath)
  converter = CharEnc(charMapList(sentences), length)
  X         = converter.textToOneHot(sentences)

  # Define the model
  in_shape  = (length, converter.embeddingLen)
  out_shape = in_shape
  m = model(in_shape, out_shape, inner_init='glorot_uniform', init='glorot_uniform')#, inner_activation='sigmoid')

  # Train the model
  m.fit(X, X, nb_epoch=nb_epoch)

  # Metrics
  y = m.predict(X[:1])
  print(converter.charMap)
  print(converter.oneHotToText(X[:1]))
  print(converter.oneHotToText(y[:1]))

_test()
