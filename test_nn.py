import numpy as np
from sklearn.datasets import load_breast_cancer
from algo.preprocess import OneHot, MinMax
from algo.nn import activation as F
from algo.nn.layers import Linear

# Preprocess
def load_data():
    data = load_breast_cancer()
    return data.data, data.target

def split_data(X, y, test_size, shuffle=True):
    assert len(X) == len(y), f"Feature has {len(X)} while label has {len(y)}"
    if shuffle :
        perm = np.random.permutation(len(X))
        X[perm], y[perm] = X, y

    sizes = int(len(X) * test_size)
    X_train, X_test = X[sizes:], X[:sizes]
    y_train, y_test = y[sizes:], y[:sizes]
    return X_train, X_test, y_train, y_test

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, shuffle=True)

sc = MinMax()
ohe = OneHot()

sc.calc(X_train)
ohe.calc(y_train)

sc_train = sc.scale(X_train)
sc_val = sc.scale(X_test)
ohe_train = ohe.scale(y_train)
ohe_val = ohe.scale(y_test)

# Model 
class Model:
    def __init__(self):
        self.x1 = Linear(X.shape[1], 32)
        self.x2 = Linear(32, 16)
        self.x3 = Linear(16, 2)

    def forward(self, inputs):
        x = F.relu(self.x1(inputs))
        print("FF1\n", x)
        print(type(x))
        x = F.relu(self.x2(x))
        print("FF2\n", x)
        print(type(x))
        return F.sigmoid(self.x3(x))

model = Model()
if __name__ == "__main__":
    y_hat = model.forward(sc_train)
    print(y_hat)
