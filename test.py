import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from algo.preprocess import Standardize
from algo.ml_supervised import KNearestNeighbor, GaussianNB, LogisticRegression, SoftmaxRegression

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

    nb = GaussianNB()
    nb.fit(sc_train, y_train)

    lr = LogisticRegression()
    print("Logistic Regression :")
    lr.fit(sc_train, y_train, data_val=(sc_test, y_test))
    print( len(lr.loss_hist), len(lr.val_loss_hist) )

    lr_no_b = LogisticRegression(use_bias=False)
    print("Logistic Regression with no bias : ")
    lr_no_b.fit(sc_train, y_train, data_val=(sc_test, y_test))
    print( len(lr_no_b.loss_hist), len(lr_no_b.val_loss_hist) )

    sr = SoftmaxRegression()
    print("Softmax Regression :")
    sr.fit(sc_train, y_train, data_val=(sc_test, y_test))
    print( len(sr.loss_hist), len(sr.val_loss_hist) )

    sr_no_b = SoftmaxRegression(use_bias=False)
    print("Softmax Regression with no bias :")
    sr_no_b.fit(sc_train, y_train, data_val=(sc_test, y_test))
    print( len(sr_no_b.loss_hist), len(sr_no_b.val_loss_hist) )

    nb_pred = nb.predict(sc_test)
    knn_pred = knn.predict(sc_test)

    lr_no_b_pred = lr_no_b.predict(sc_test)
    lr_pred = lr.predict(sc_test)

    sr_no_b_pred = sr_no_b.predict(sc_test)
    sr_pred = sr.predict(sc_test)

    print("NB Accuracy : ", accuracy(y_test, nb_pred))
    print("KNN Accuracy : ", accuracy(y_test, knn_pred))
    print("LR with no bias Acc : ", accuracy(y_test, lr_no_b_pred))
    print("LR with bias Acc : ", accuracy(y_test, lr_pred))
    print("SR with no bias Acc : ", accuracy(y_test, sr_no_b_pred))
    print("SR with bias Acc : ", accuracy(y_test, sr_pred))
