import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from keras_preprocessing.text import hashing_trick
import itertools as it

class Generator:

    def __init__(self):
        self.data = []
        self.dict = []

    def window(self, arr, window=3):
        if(window > len(arr)): return None

        windows = []
        labels = []

        for idx in range(len(arr) - window - 1):
            windows.append(arr[idx:idx+window])
            labels.append(arr[idx+window])

        return windows, labels


    def generate_data(self, path, window_size=15):
        f = open(path, 'r')

        sonnet_names = ['C',
                        'CI',
                        'CII',
                        'CIII',
                        'CIV',
                        'CIX',
                        'CL',
                        'CLI',
                        'CLII',
                        'CLIII',
                        'CLIV',
                        'CV',
                        'CVI',
                        'CVII',
                        'CVIII',
                        'CX',
                        'CXI',
                        'CXII',
                        'CXIII',
                        'CXIV',
                        'CXIX',
                        'CXL',
                        'CXLI',
                        'CXLII',
                        'CXLIII',
                        'CXLIV',
                        'CXLIX',
                        'CXLV',
                        'CXLVI',
                        'CXLVII',
                        'CXLVIII',
                        'CXV',
                        'CXVI',
                        'CXVII',
                        'CXVIII',
                        'CXX',
                        'CXXI',
                        'CXXII',
                        'CXXIII',
                        'CXXIV',
                        'CXXIX',
                        'CXXV',
                        'CXXVI',
                        'CXXVII',
                        'CXXVIII',
                        'CXXX',
                        'CXXXI',
                        'CXXXII',
                        'CXXXIII',
                        'CXXXIV',
                        'CXXXIX',
                        'CXXXV',
                        'CXXXVI',
                        'CXXXVII',
                        'CXXXVIII',
                        'L',
                        'LI',
                        'LII',
                        'LIII',
                        'LIV',
                        'LIX',
                        'LV',
                        'LVI',
                        'LVII',
                        'LVIII',
                        'LX',
                        'LXI',
                        'LXII',
                        'LXIII',
                        'LXIV',
                        'LXIX',
                        'LXV',
                        'LXVI',
                        'LXVII',
                        'LXVIII',
                        'LXX',
                        'LXXI',
                        'LXXII',
                        'LXXIII',
                        'LXXIV',
                        'LXXIX',
                        'LXXV',
                        'LXXVI',
                        'LXXVII',
                        'LXXVIII',
                        'LXXX',
                        'LXXXI',
                        'LXXXII',
                        'LXXXIII',
                        'LXXXIV',
                        'LXXXIX',
                        'LXXXV',
                        'LXXXVI',
                        'LXXXVII',
                        'LXXXVIII',
                        'X',
                        'XC',
                        'XCI',
                        'XCII',
                        'XCIII',
                        'XCIV',
                        'XCIX',
                        'XCV',
                        'XCVI',
                        'XCVII',
                        'XCVIII',
                        'XI',
                        'XII',
                        'XIII',
                        'XIV',
                        'XIX',
                        'XL',
                        'XLI',
                        'XLII',
                        'XLIII',
                        'XLIV',
                        'XLIX',
                        'XLV',
                        'XLVI',
                        'XLVII',
                        'XLVIII',
                        'XV',
                        'XVI',
                        'XVII',
                        'XVIII',
                        'XX',
                        'XXI',
                        'XXII',
                        'XXIII',
                        'XXIV',
                        'XXIX',
                        'XXV',
                        'XXVI',
                        'XXVII',
                        'XXVIII',
                        'XXX',
                        'XXXI',
                        'XXXII',
                        'XXXIII',
                        'XXXIV',
                        'XXXIX',
                        'XXXV',
                        'XXXVI',
                        'XXXVII',
                        'XXXVIII']

        text = f.read()
        f.close()

        words = word_tokenize(text)
        words = [word for word in words if word not in sonnet_names]
        words = [word.lower() for word in words]
        dictionary = sorted(list(set(words)))
        self.dict = {}
        for idx, word in enumerate(dictionary):
            self.dict[word] = idx

        encoded_list = []

        for word in words:
            encoded_list.append(self.dict[word])

        windows, data = self.window(encoded_list, window=window_size)
        print(encoded_list[0:20])
        print(windows[0:20])
        print(data[0:20])
        print(self.dict)

        return np.array(windows), np.array(data), self.dict
