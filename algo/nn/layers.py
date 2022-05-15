import numpy as np

class Linear :
    def __init__(self, inNodes : int, outNodes : int, use_bias : bool =True, init : str = "normal"):
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.use_bias = use_bias
        self.init = init.lower()
        if self.use_bias : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes))
                b = np.random.normal(loc=0., scale=0.05, size=((self.outNodes)))
                self.w, self.b = w, b
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes))
                b = np.random.uniform(low=-0.05, high=0.05, size=((self.outNodes)))
                self.w, self.b = w, b
            elif self.init == "he_normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)
                b = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)
                self.w, self.b = w, b
            else :
                raise ValueError("Weights initializer is not valid")
        else : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes))
                self.w = w
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes))
                self.w = w
            elif self.init == "he_normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)
                self.w = w
            else :
                raise ValueError("Weights initializer is not valid")

    def forward(self, inputs):
        if self.use_bias:
            self.out = np.matmul(inputs, self.w) + self.b
        else :
            self.out = np.matmul(inputs, self.w)

    def backward(self):
        pass

class Flatten:
    def __init__(self, inNodes : int, outNodes : int): # COMMAND : input nodes and output nodes is later used for backpropagation
        self.inNodes = inNodes
        self.outNodes = outNodes

    def forward(self, inputs): 
        self.out = np.reshape(inputs, newshape=(-1, 1))

    def backward(self):
        # np.reshape(inputs, newshape=(self.inNodes, self.outNodes))

