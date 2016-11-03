from os import path

import numpy as np

from dataencoders import *
from dataencoders import std

BareSentencesPath = '../data/processed/en-ud.txt'
EnglishUT         = "../data/Universal Dependencies 1.3/ud-treebanks-v1.3/UD_English"
EnglishUT_dev     = path.join(EnglishUT, 'en-ud-train.conllu')

def loadLines(fname): # String => Matrix[X :: HNil, String]
  with open(fname) as f:
    content = f.readlines()
    return np.array(content, dtype=object)

def load(fname): # String => String
  with open(fname) as f:
    return f.read()

# String => Matrix[Sentences :: sent_size :: HNil, String]
def string_to_sentences(str, sent_size=20, pad=''): return process_array_by_elem(np.array([str], dtype=object), [
  std.string_splitter('\n\n', identity)  # Matrix[1 :: Sentences :: HNil, String]
, std.lift_first_dim                     # Matrix[Sentences :: HNil, String]
, std.lines(std.Padding(sent_size, pad)) # Matrix[Sentences :: sent_size :: HNil, String]
])

# Matrix[Sentences :: Words :: HNil, WordWithTags] => Matrix[Sentences :: Words :: HNil, Word]
def sentences_to_words(sents, delim='\t', pad=''): return process_array_by_elem(sents, [
  std.string_splitter(delim, identity)  # Matrix[Sentences :: Words :: Tokens :: HNil, String]
, std.filter_indices([1])               # Matrix[Sentences :: Words :: HNil, String]
])

# Matrix[Sentences :: Words :: HNil, Word] => Matrix[Sentences :: Words :: Chars :: HNil, Char]
def words_to_chars(words, length=10, pad=''): return process_array_by_elem(words, [
  std.chars(std.Padding(length, pad))
])

# Matrix[Sentences :: Words :: HNil, Char] => Matrix[Sentences :: Chars :: HNil, Char]
def words_to_raw_chars(words, length=50, pad=''): return process_array_by_elem(words, [
  lambda word: np.array([word, ' '], object)
, std.flatten_last_dim
, std.chars(std.Padding(20, ''))
, std.flatten_last_dim
, std.filter(identity, std.Padding(length, pad))
])


### TESTING ###

def chain(inp, methods):
  res = inp
  for m in methods:
    res = m(res)
  return res

def _ds(): return load(EnglishUT_dev)

def _test_string_to_sentences():
  arr = string_to_sentences(_ds())
  print(arr[:5])

def _test_sentences_to_words():
  arr = chain(_ds(), [string_to_sentences, sentences_to_words])
  print(arr[:5])

def _test_words_to_chars():
  arr = chain(_ds(), [string_to_sentences, sentences_to_words, words_to_chars])
  print(arr[:5])

def _test_words_to_raw_chars():
  arr = chain(_ds(), [string_to_sentences, sentences_to_words, words_to_raw_chars])
  print(arr[:5])

_test_words_to_raw_chars()
