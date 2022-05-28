from collections import defaultdict
import numpy as np

class Tensor(object):
    def __init__(self):
        pass

class Linear :
    def __init__(self, inNodes : int = None,  outNodes : int = None,  use_bias : bool = True,  init : str = "normal", dtype=np.float32):
        self.inNodes = inNodes
        self.dtype = dtype
        self.outNodes = outNodes
        self.use_bias = use_bias
        self.init = init.lower()
        self.params = {}
        self.__initialize()
        self.__set_name()
        self.__make_dict()

    def __call__(self, inputs):
        if self.use_bias:
            z = np.matmul(inputs, self.w) + self.b
        else :
            z = np.matmul(inputs, self.w)
        self.z = z
        return self.z

    def __set_name(self):
        self.name = f"linear_{self.outNodes}"

    def __initialize(self):
        if self.use_bias : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)).astype(self.dtype)
                b = np.random.normal(loc=0., scale=0.05, size=((self.outNodes))).astype(self.dtype)
                self.w, self.b = w, b
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes)).astype(self.dtype)
                b = np.random.uniform(low=-0.05, high=0.05, size=((self.outNodes))).astype(self.dtype)
                self.w, self.b = w, b
            elif self.init == "he_normal":
                w = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)).astype(self.dtype)
                b = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)).astype(self.dtype)
                self.w, self.b = w, b
            else :
                raise ValueError("Weights initializer is not valid")
        else : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)).astype(self.dtype)
                self.w = w
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes)).astype(self.dtype)
                self.w = w
            elif self.init == "he_normal":
                w = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)).astype(self.dtype)
                self.w = w
            else :
                raise ValueError("Weights initializer is not valid")

    def backward(self, inputs):
        if self.use_bias:
            dW = np.matmul(self.z.T, inputs)
            db = np.sum(self.z, axis=0, keepdims=True)
        else :
            dW = np.matmul(self.z.T, inputs)

    def __make_dict(self):
        if self.use_bias :
            param = { self.name : { "W" : self.w, "b" : self.b } }
            self.params.update(param)
        else : 
            param = { self.name : { "W" : self.w } }
            self.params.update(param)

class Flatten:
    def __init__(self, inNodes : int, outNodes : int): # COMMAND : input nodes and output nodes is later used for backpropagation
        self.inNodes = inNodes
        self.outNodes = outNodes

    def __call__(self, inputs): 
        self.out = np.reshape(inputs, newshape=(-1, 1))

    def backward(self):
        self.out = np.reshape(inputs, newshape=(self.inNodes, self.outNodes))
