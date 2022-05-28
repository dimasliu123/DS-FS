import numpy as np


class BinaryCrossEntropy:
    def __init__(self, epsilon : float = 1e-7, reduction : str = "mean"):
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
    def __init__(self, epsilon):
        self.epsilon = epsilon

