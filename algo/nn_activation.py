import numpy as np
# Neural Network activation 

class Activation:
	def __init__(self, z):
		self.z = z

class ReLU(Activation):
	def __init__(self, z):
		super().__init__(z)

	def forward(self): # f(x) = {z <= 0, z > 0 = z}
		z = self.z
		return np.maximum(0, z)

	def backward(self): # f'(x) = = {z <= 0 = 0, z > 0 = 1}
		z = self.z
		z[z<=0] = 0
		z[z>0] = 1
		return z

class Sigmoid(Activation):
	def __init__(self, z):
		super().__init__(z)

	def forward(self): # f(x)  = 1 / ( 1 + e^-x)
		z = self.z
		return 1 / (1 + np.exp(-z))

	def backward(self): # f'(x) = f(x) - (1 - f(x))
		z = self.z
		return self.forward() - ( 1 - self.forward())

class Tanh(Activation):
	def __init__(self, z):
		super().__init__(z)

	def forward(self): # f(x) = (e^x - e^-x) / (e^x + e^-x)
		z = self.z
		return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) 

	def backward(self): # f'(x) = 1 - f(x)^2 	
		z = self.z
		return 1. - self.forward(z) ** 2

class Softmax(Activation):
    def __init__(self, z):
        super().__init__(z)
    
    def forward(self):
        z = self.z
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    def backward(self):
        z = self.z
        z_flat = z.reshape(-1, 1)
        z_diag = np.diagflat(z)
        return z_diag - np.dot(z_flat, z_flat.T)
