import numpy as np

class Otimizer :
    def __init__(self, lr : float):
        self.lr = lr 

class SGD(Optimizer):
    def __init__(self, lr = 0.01, momentum : float = None):
        super().__init__(lr)
        self.momentum = momentum
        if self.momentum is not None:
            assert self.momentum < 1.0, f"Momentum can't be larger than 1.0 during the gradient updates, while momentum has {self.momentum}"

    def update():
        if self.use_bias :
            if self.momentum is not None:
                pass
            else :
                w = w - self.lr * dw 
                b = b - self.lr * db
        else : 
            if self.momentum is not None :
                pass
            else :
                w = w - self.lr * dw

class AdaGrad(Optimizer):
    def __init__(self, lr = 0.01, epsilon : float = 1e-6):
        super().__init(lr)
        self.epsilon = epsilon

    def update():
        if self.use_bias :
            w_nota = np.dot(self.lr / (dw + self.epsilon), dw)
            b_nota = np.dot(self.lr / (dw + self.epsilon), db)
            w = w - w_nota
            b = b - b_nota 
        else : 
            w_nota = np.dot(self.lr / (dw + self.epsilon), dw)
            w = w - w_nota 

class AdaDelta(Optimizer):
    def __init__(self, lr = 0.01): # NOTE : RMS = (1 / n_batch)  * sum(xi**2)
        super().__init(lr)
        pass
