import numpy as np
from sklearn.datasets import load_breast_cancer
from algo.preprocess import OneHot, MinMax
from algo.nn import activation as F
from algo.nn.loss import BinaryCrossEntropy
from algo.nn.neur_nets import NeuralNets

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

def main():
    bce = BinaryCrossEntropy()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.10, shuffle=True)

    sc = MinMax()
    ohe = OneHot()
    sc.calc(X_train)
    ohe.calc(y_train)

    sc_train = sc.scale(X_train)
    sc_val = sc.scale(X_test)
    ohe_train = ohe.scale(y_train)
    ohe_val = ohe.scale(y_test)
    model = NeuralNets()
    y_hat = model.forward(sc_train)
    loss = bce(ohe_train, y_hat)

    print(loss)
    print(y_hat[0:2])

if __name__ == "__main__":
    main()
