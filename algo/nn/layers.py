from collections import defaultdict
import numpy as np

"""
TODO :
getattr(Linear.use_bias)
params = defaultdict() 
for i in MODEL:
    if getattr(Linear.use_bias) is True:
        params.update(Linear.b)
    params.update(Linear.w)
pseudo-code :
class Module:
    def __init__(self):
        self.x1 = Linear(32, 16, use_bias=True)
        self.x2 = Linear(16, 8, use_bias=True)
        self.out = Linear(8, 2, use_bias=True)

    def forward(self, inputs): -> matmul
        x = self.x1(inputs)
        x = self.x2(x)
        return self.out(x)
"""
class Linear :
    def __init__(self, inNodes : int,  outNodes : int,  use_bias : bool = True,  init : str = "normal", dtype=np.float32):
        self.inNodes = inNodes
        self.dtype = dtype
        self.outNodes = outNodes
        self.use_bias = use_bias
        self.init = init.lower()

        if self.use_bias : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)).astype(dtype)
                b = np.random.normal(loc=0., scale=0.05, size=((self.outNodes))).astype(dtype)
                self.w, self.b = w, b
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes)).astype(dtype)
                b = np.random.uniform(low=-0.05, high=0.05, size=((self.outNodes))).astype(dtype)
                self.w, self.b = w, b
            elif self.init == "he_normal":
                w = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)).astype(dtype)
                b = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)) * np.sqrt(2 / self.inNodes)).astype(dtype)
                self.w, self.b = w, b
            else :
                raise ValueError("Weights initializer is not valid")
        else : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes)).astype(dtype)
                self.w = w
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=(self.inNodes, self.outNodes)).astype(dtype)
                self.w = w
            elif self.init == "he_normal":
                w = (np.random.normal(loc=0., scale=0.05, size=(self.inNodes, self.outNodes), dtype=dtype) * np.sqrt(2 / self.inNodes)).astype(dtype)
                self.w = w
            else :
                raise ValueError("Weights initializer is not valid")

    def __call__(self, inputs):
        if self.use_bias:
            z = np.matmul(inputs, self.w) + self.b
        else :
            z = np.matmul(inputs, self.w)
        self.z = z
        return self.z

class Flatten:
    def __init__(self, inNodes : int, outNodes : int): # COMMAND : input nodes and output nodes is later used for backpropagation
        self.inNodes = inNodes
        self.outNodes = outNodes

    def __call__(self, inputs): 
        self.out = np.reshape(inputs, newshape=(-1, 1))

    def backward(self):
        self.out = np.reshape(inputs, newshape=(self.inNodes, self.outNodes))
