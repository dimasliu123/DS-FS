import numpy as np

class Linear :
    def __init__(self, 
                 inNodes : int, 
                 outNodes : int, 
                 use_bias : bool =True,
                 w_init : str = "normal"):
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.use_bias = use_bias
        self.w_init = w_init.lower()
        
    def init_weights(self):
        if self.use_bias : 
            if self.w_init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes))
                b = np.random.normal(loc=0., scale=0.05, size=((self.outNodes)))
                self.w, self.b = w, b
            elif w_init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes))
                b = np.random.uniform(low=-0.05, high=0.05, size=((self.outNodes)))
                self.w, self.b = w, b
            else :
                raise ValueError("Weights initializer is not valid")
            return self.w, self.b
        else : 
            if self.w_init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes))
                self.w = w
            elif w_init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes))
                self.w = w
            else :
                raise ValueError("Weights initializer is not valid")
            return self.w
