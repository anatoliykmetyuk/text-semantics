from keras.layers import *
from keras.models import Model, Sequential

import numpy as np

dim           = 2
samples_size  = 10
length        = 1
nb_epoch      = 4000

def model(in_shape, out_shape, **kwargs):
  l = SimpleRNN(
      output_dim      = out_shape[-1]
    , input_dim        = in_shape[-1]
    , input_length     = in_shape[0]
    , return_sequences = True
    , **kwargs)

  model = Sequential()
  model.add(l)

  model.compile(loss='mse', optimizer='sgd')

  print(model.summary())
  return model

def _test():
  in_shape  = (length, dim)
  out_shape = in_shape
  features  = samples_size * in_shape[0] * in_shape[1]
  m = model(in_shape, out_shape)#, init = initializations.identity, inner_init = initializations.zero)

  sample      = np.arange(samples_size * in_shape[0] * in_shape[1], dtype=np.float64).reshape(samples_size, in_shape[0], in_shape[1]) / features
  sample_test = np.random.random((10, in_shape[0], in_shape[1]))
  # np.arange(delta, delta + samples_size * in_shape[0] * in_shape[1], dtype=np.float64).reshape(samples_size, in_shape[0], in_shape[1]) / features + 

  m.fit(sample, sample, nb_epoch=nb_epoch)
  print(m.get_weights())

  def eval(s, name, take=10):
    print(name)
    print("Sample:\n", s[:take])
    print("Prediction:\n", m.predict(s)[:take])

  eval(sample, "Train")
  # eval(sample_test, "Test")


_test()

# from recurrentshop import*
# from keras.layers import*
# from keras.models import*
# import numpy as np

# # Script for comparing performance of native keras and recurrentshop stacked RNN implementations
# # We observe 20-30% speed ups on GPU


# import sys
# sys.setrecursionlimit(10000000)

# # Params

# input_length = 10
# dim = 1
# nb_epoch = 5
# unroll = False

# # Random data

# x = np.random.random((10, input_length, dim))
# y = np.random.random((10, dim))

# # recurrentshop model

# rc = RecurrentContainer(input_length=input_length, unroll=unroll)
# rc.add(LSTMCell(dim, input_dim=dim))

# model = Sequential()
# model.add(rc)

# model.compile(loss='mse', optimizer='sgd')
# print(model.summary())

# print('Compiling...')
# model.train_on_batch(x[:1], y[:1])  # force compile

# model.fit(x, y, nb_epoch=nb_epoch)
