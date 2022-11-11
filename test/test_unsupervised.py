import numpy as np
from algo.datasets import load_mnist
from algo.ml.unsupervised import RBM

def main():
    X_train, y_train, X_test, y_test = load_mnist(use_scale=True)
    del y_train, y_test

    data_shape = X_train.shape[1] * X_train.shape[2]

    sc_train = X_train.reshape(-1, data_shape)
    sc_test = X_test.reshape(-1, data_shape)

    rbm = RBM(n_visible=data_shape)
    rbm.fit(sc_train, batch_size = 65, data_val = sc_test, steps=100)

if __name__ == "__main__":
    main()
