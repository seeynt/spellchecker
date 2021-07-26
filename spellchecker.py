import re
import numpy
import gzip
import pickle
from collections import Counter
from random import uniform
from collections import defaultdict
from math import log
from sys import stdin
from functools import reduce

class StandardTrie:
    def __init__(self, strings):
        self._root = self.Node()

        for s in strings:
            self.add(s, s)
            
    @property
    def root(self):
        return self._root

    def add(self, string, value):
        node = self._root
        for char in string:
            found_in_child = False
            for child in node.children:
                if child.key == char:
                    found_in_child = True
                    node = child
                    break
            if not found_in_child:
                new_node = self.Node()
                new_node.key = char
                node.children[new_node] = new_node
                node = new_node
        node._value = value

    def find(self, pattern):
        node = self._root

        if not node._children:
            return False

        for char in pattern:
            char_not_found = True
            for child in node._children:
                if child.key == char:
                    char_not_found = False
                    node = child
                    break
            if char_not_found:
                return False

        if not node._value:
            return False
        else:
            return node._value
    
    class Node:
        __slots__ = '_children', '_key', '_value'

        def __init__(self):
            self._children = {}
            self._key = None
            self._value = None

        @property
        def children(self):
            return self._children

        @property
        def key(self):
            return self._key

        @key.setter
        def key(self, new_key):
            self._key = new_key

        @property
        def value(self):
            return self._value

class TrieSpellChecker:
    def __init__(self, lexicon):
        self._lexicon = StandardTrie(lexicon)
        self._threshold = 5

    def check(self, word):
        spellings = []

        if self._lexicon.find(word):
            spellings.append(word)

        root_node = self._lexicon.root

        current_row = range(len(word) + 1)

        for child in root_node.children:
            self._recursive_check(child, child.key, word,
                                 current_row, spellings)

        if not spellings:
            return [word]
        
        return spellings

    def _recursive_check(self, node, char, word, previous_row, spellings):
        num_cols = len(word) + 1
        current_row = [previous_row[0] + 1]

        # levenstein
        for col in range(1, num_cols):

            left = current_row[col - 1] + 1
            up = previous_row[col] + 1
            diagonal = previous_row[col - 1]

            if word[col - 1] != char:
                diagonal += 1

            current_row.append(min(left, up, diagonal))

        if current_row[-1] <= self._threshold and node.value != None:
            spellings.append(node.value)

        if min(current_row) <= self._threshold:
            for child in node.children:
                self._recursive_check(child, child.key, word,
                                     current_row, spellings)

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
        return reduce((lambda x, y: x * y), list(map(self.P, text.split())), 1)
        
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

class SpellChecker:
    def __init__(self, language_model):
        self._language = language_model
        self._words = self._language.corpus()
        self._trie = TrieSpellChecker(list(self._words.keys()))
        self._threshold = 1e-7
    
    def _correct_word(self, word):
        candidates = self._trie.check(word)
        if word in candidates and self._language.P(word) > self._threshold:
            return word
        return max(self._trie.check(word), key=self._language.P)
    
    def correct(self, text):
        words = text.lower().split()
        words = list(map(self._correct_word, words))
        result = ' '.join(words)
        if result != text:
            return True, result
        return False, result

class LayoutChecker:
    def __init__(self, language_model):
        self._language = language_model
        self._eng_chars = " ~!@#$%^&qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP{}ASDFGHJKL:\"|ZXCVBNM<>?"
        self._rus_chars = " ё!\"№;%:?йцукенгшщзхъфывапролджэячсмитьбю.ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭ/ЯЧСМИТЬБЮ,"
        self._eng_to_rus = dict(zip(self._eng_chars, self._rus_chars))
        self._rus_to_eng = dict(zip(self._rus_chars, self._eng_chars))

    def correct(self, text):
        text_rus = ''.join([self._eng_to_rus.get(c, c) for c in text])
        text_eng = ''.join([self._rus_to_eng.get(c, c) for c in text])
        result = max([text, text_rus, text_eng], key=self._language.proba)   
        if result != text:
            return True, result
        return False, result  

class SplitChecker:
    def __init__(self, language_model):
        self._language = language_model
        self._maxword = max(len(w) for w in self._language.corpus())
        
    def correct(self, text):
        l = [self._split(x) for x in text.split()]
        result = ' '.join([item for sublist in l for item in sublist])
        if result != text:
            return True, result
        return False, result
    
    def _split(self, s):
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self._maxword):i]))
            return min((c - log(self._language.P(s[i - k - 1:i].lower())), k + 1) for k, c in candidates)

        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            out.append(s[i - k:i])
            i -= k

        return reversed(out)

class JoinChecker:
    def __init__(self, language_model):
        self._language = language_model
    
    def correct(self, text):
        i = 0
        s = text
        while s.find(' ', i + 1) != -1:
            i = text.find(' ', i + 1)
            if self._language.proba(s[0:i] + s[i + 1:]) > self._language.proba(s):
                s = s[0:i] + s[i + 1:]
        
        if s != text:
            return True, s
        return False, s

if __name__ == "__main__":
    language_model = LanguageModel('spellchecker')

    spellchecker = SpellChecker(language_model)
    layoutchecker = LayoutChecker(language_model)
    splitchecker = SplitChecker(language_model)
    joinchecker = JoinChecker(language_model)

    for line in stdin:
        if (line == '\n'):
            break
        flag, line = layoutchecker.correct(line)
        flag, line = splitchecker.correct(line)
        flag, line = joinchecker.correct(line)
        flag, line = spellchecker.correct(line)

        print(line)
