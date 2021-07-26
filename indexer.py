import re
import numpy
import gzip
import pickle
from collections import Counter
from random import uniform
from collections import defaultdict
from math import log
from functools import reduce

class LanguageModel:
    def __init__(self, flag = 'all'):
        self._model= {}
        self.N = 0
        text = ''
        
        if flag == 'indexer' or flag == 'all':
            for line in self._gen_lines('queries_all.txt'):
                parts = line.split('\t')
                if len(parts) == 1:
                    text += line + ' '
                if len(parts) == 2:
                    text += parts[1] + ' '
            
            words_counter = Counter(self._words(text))
            self._model = words_counter
            
        if flag == 'indexer':
            self._save()
            
        if flag == 'spellchecker':
            self._load()
            
        self.N = sum(self._model.values())
        
    def corpus(self):
        return self._model
    
    def P(self, word):
        return self._model.get(word, 0.1) / self.N
    
    def proba(self, text):
        return reduce((lambda x, y: x * y), list(map(self.P, text.split())))
        
    def _words(self, text): 
        return re.findall(r'\w+', text.lower())
    
    def _gen_lines(self, fname):
        with open(fname) as data:
            for line in data:
                yield line.lower()
            
    def _save(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump(self._model, f, pickle.HIGHEST_PROTOCOL)

    def _load(self):
        with open('model.pkl', 'rb') as f:
            self._model = pickle.load(f)

if __name__ == "__main__":
    language_model = LanguageModel('indexer')