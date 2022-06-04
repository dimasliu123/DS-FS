import numpy as np


class BinaryCrossEntropy:
    def __init__(self, epsilon : float = 1e-8, reduction : str = "mean"):
        self.epsilon = epsilon 
        self.reduction = reduction.lower()

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, a_min=self.epsilon, a_max = 1 - self.epsilon)
        nota1 = y_true * np.log(y_pred + self.epsilon)
        nota2 = ( 1 - y_true ) * np.log( 1 - y_pred + self.epsilon )
        nota = nota1 + nota2
        self.loss = - np.mean(nota)
        return self.loss

class CategoricalCrossEntropy:
    def __init__(self, epsilon : float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, a_min = self.epsilon, a_max = 1 - self.epsilon)
        return  - np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]) + self.epsilon)

class KLDivergence:
    def __init__(self, epsilon : float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        y_true = np.clip(y_true, a_min=self.epsilon, a_max = 1.)
        y_true = np.clip(y_true, a_min=self.epsilon, a_max = 1.)
        return np.sum(y_true * np.log(y_true / y_pred)) 
        
