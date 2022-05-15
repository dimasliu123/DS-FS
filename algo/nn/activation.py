import numpy as np

class ReLU:
	def forward(self, inputs): # f(x) = {inputs <= 0, z > 0 = z}
		self.out = np.maximum(0, inputs)

	def backward(self, inputs): # f'(x) = = {inputs <= 0 = 0, z > 0 = 1}
		inputs[inputs<=0] = 0
		inputs[inputs>0] = 1
		return inputs

class Sigmoid:
	def forward(self, inputs): # f(x)  = 1 / ( 1 + e^-x)
		self.out = 1 / (1 + np.exp(-inputs))

	def backward(self): # f'(x) = f(x) - (1 - f(x))
		return self.forward() - ( 1 - self.forward())

class Tanh:
	def forward(self, inputs): # f(x) = (e^x - e^-x) / (e^x + e^-x)
		self.out = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs)) 

	def backward(self): # f'(x) = 1 - f(x)^2 	
		self.out = 1. - self.forward(inputs) ** 2

class Softmax:
    def forward(self, inputs): # f(x) = e(x) / ∑e(x)
        exp_inputs = np.exp(inputs)
        self.out = exp_inputs / np.sum(exp_z)
    
    def backward(self):
        inputs_flat = z.reshape(-1, 1)
        inputs_diag = np.diagflat(z)
        return inputs_diag - np.dot(z_flat, z_flat.T)
