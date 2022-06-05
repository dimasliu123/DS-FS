import numpy as np

def load_mnist(use_scale : bool = False):
    scale = lambda x : x / 255.0

    X_train = np.load("algo/data/MNIST/feat_train.npy")
    X_test = np.load("algo/data/MNIST/feat_test.npy")
    y_train = np.load("algo/data/MNIST/label_train.npy")
    y_test = np.load("algo/data/MNIST/label_test.npy")

    if use_scale :
        X_train = scale(X_train).astype(np.float32)
        X_test = scale(X_test).astype(np.float32)
        y_train = scale(y_train).astype(np.float32)
        y_test = scale(y_test).astype(np.float32)
        return X_train, y_train, X_test, y_test
    else : 
        return X_train, y_train, X_test, y_test
