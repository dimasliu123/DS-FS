import numpy as np

def relu(x, back=False):  # f(x) = (inputs <= 0, z > 0 = z)
    if back: # f'(x) = (inputs <= 0 = 0, z > 0 = 1)
        x[x <= 0] = 0.
        x[x > 0] = 1.
        return x
    return np.maximum(0, x)

def sigmoid(x, back=False): # f(x)  = 1 / ( 1 + e^-x)
    def forward():
        return 1 / ( 1 + np.exp(-x))
    if back : # f'(x) = f(x) - (1 - f(x))
        return forward() - (1 - forward())
    return forward()


class NeuralNets :
    def __init__(self):
        self.w1 = np.random.uniform( low=-0.05, high=0.05, size=(30, 32) ) 
        self.w2 = np.random.uniform( low=-0.05, high=0.05, size=(32, 8) )
        self.out = np.random.uniform( low=-0.05, high=0.05, size=(8, 2) )

    def forward(self, inputs):
        self.z1 = np.matmul(inputs, self.w1)
        self.a1 = relu(self.z1)
        self.z2 = np.matmul(self.a1, self.w2)
        self.a2 = relu(self.z2)
        self.z3 = np.matmul(self.a2, self.out)
        self.y_hat = sigmoid(self.z3)
        return self.y_hat

    @staticmethod
    def __calcGrad(X1, X2):
        return np.matmul(X1.T, X2)
