from keras.layers import *
from keras.models import Model, Sequential

import numpy as np

from util import dataset
from dataencoders import CharEnc, charMapList

from framework import *

import argparse

# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which model to use: lstm, rnn or dense")
parser.add_argument("-d", "--data" , help="Which data to train on: one, two, many")
args = parser.parse_args()


# Helpers
def one_string(which):
  strings_identity(which, nb_epoch=1000, sentences=['ASDFGHJKQWERTYYIUOSSSHTHDFSGARSEGSERGSERHRSTHRSTHRTSH'])

def one_string_padding(which):
  strings_identity(which, nb_epoch=1000, sentences=['ABC', 'DFHSMNG'])

def many_strings(which):
  print("Loading the data")
  sentences = dataset.loadLines(dataset.BareSentencesPath)
  print("Loaded")
  strings_identity(which, nb_epoch=25, sentences=sentences)


# Config
# See: http://stackoverflow.com/a/60211/3895471
def resolve_model(name): return {
  'lstm' : lstm
, 'rnn'  : rnn
, 'dense': dense_first
}[name]

def resolve_data(name): return {
  'one' : one_string
, 'two' : one_string_padding
, 'many': many_strings
}[name]

# Run
objective = resolve_data (args.data)
mdl       = resolve_model(args.model)
objective(mdl)
