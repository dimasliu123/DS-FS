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

def tanh(x, back=False): # f(x) = (e^x - e^-x) / (e^x + e^-x)
    if back : # f'(x) = 1 - f(x)^2
        return np.tanh(x)
    return 1. - np.tanh(x) ** 2

def softmax(x, back=False):
    if back: # f(x) = e(x) / ∑e(x)
        nota1 = x / np.sum(x, axis=0)
        nota2 = ( 1 - x / np.sum(x, axis=0))
        return nota1 * nota2
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()
