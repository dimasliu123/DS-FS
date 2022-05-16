import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from algo.preprocess import Standardize
from algo.ml_supervised import KNearestNeighbor, NaiveBayes, LogisticRegression 

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true) * 100

def load_data():
    data = load_breast_cancer()
    return data.data, data.target

def shuffle(X, y):
    assert len(X) == len(y), f"Feature has {len(X)} while label has {len(y)}"
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]

def split_data(X, y, test_size):
    sizes = int(len(X) * test_size)
    X_train, X_test = X[sizes:], X[:sizes]
    y_train, y_test = y[sizes:], y[:sizes]
    return X_train, X_test, y_train, y_test


# Load ML
if __name__ == "__main__":
    X, y = load_data()
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.15)
    print(X_train.shape, y_train.shape)

    sc = Standardize()
    sc.calc(X_train)
    sc_train = sc.scale(X_train)
    sc_test = sc.scale(X_test)

    del X_train, X_test

    knn = KNearestNeighbor()
    knn.fit(sc_train, y_train)

    nb = NaiveBayes()
    nb.fit(sc_train, y_train)

    lr = LogisticRegression()
    lr.fit(sc_train, y_train)

    lr_no_b = LogisticRegression(use_bias=False)
    lr_no_b.fit(sc_train, y_train)

    nb_pred = nb.predict(sc_test)
    knn_pred = knn.predict(sc_test)
    lr_no_b_pred = lr_no_b.predict(sc_test)
    lr_pred = lr.predict(sc_test)

    print("NB Accuracy : ", accuracy(y_test, nb_pred))
    print("KNN Accuracy : ", accuracy(y_test, knn_pred))
    print("LR with no bias Acc : ", accuracy(y_test, lr_no_b_pred))
    print("LR with bias Acc : ", accuracy(y_test, lr_pred))
