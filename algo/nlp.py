import numpy as np

class Tokenizer :
    def __init__(self, text : str, char_level : bool =False, lower : bool = True):
        self.char_level = char_level
        self.lower = lower

    def fit(self, X):
    
