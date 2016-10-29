import json
import codecs
from random import shuffle

BareSentencesPath = '../data/processed/en-ud.dev.txt'

def loadLines(fname):
  with open(fname) as f:
    content = f.readlines()
    return content
