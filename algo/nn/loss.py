import numpy as np

class Loss :
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

class BinaryCrossEntropy(Loss):
    def __init__(self, clipvalue : float = None):
        super().__init(y_true, y_pred)
        self.clipvalue = clipvalue

    def calculate_loss(self):
        if self.clipvalue is not None:
            y_pred = np.clip(self.y_pred, a_min=self.clipvalue, a_max = 1 - self.clipvalue)
            nota1 = self.y_true * np.log(y_pred + self.clipvalue)
            nota2 = ( 1 - self.y_true ) * np.log( 1 - self.y_true + self.clipvalue )
            nota = nota1 / nota2
            self.loss = -np.mean(nota)
        else :
            nota1 = self.y_true * np.log(self.y_pred)
            nota2 = (1 - self.y_true ) * np.log(1 - self.y_true)
            nota = nota1 / nota2
            self.loss = - np.mean(nota)

class CategoricalCrossEntropy(Loss):
    def __init__(self, clipvalue):
        super().__init__(y_true, y_pred)
        self.clipvalue = clipvalue

