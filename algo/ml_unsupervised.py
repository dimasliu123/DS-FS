import numpy as np

class PCA : 
    def __init__(self, n_components : int):
        self.n_components = n_components

    def fit(self, X):
        mu = np.mean(X, axis=0)
        X_mu = X - mu
        self.mu = mu
        covMat = np.cov(X_mu.T)
        eigVal, eigVec = np.linalg.eig(covMat)
        eigVec = eigVec.T
        idxs = eigVec.argsort()[::-1]
        eigVal, eigVec = eigVal[idxs], eigVec[idxs]
    
        self.components = eigVec[0 : self.n_components]

    def transform(self, X):
        X = X - self.mu
        return np.dot(X, self.components.T)

class KMeans :
    def __init__(self, n_clusters : int):
        self.n_clusters = n_clusters

    def fit(self, X):
        X = np.array(X)
        self.X = X
        return self

    def findCentroid(self, X):
        return 

    @staticmethod
    def euclidean_distance(P, Q):
        dist = 0.
        for i in range(len(P)):
            dist += np.sqrt((P[i] - Q[i]) ** 2)
        return dist

class LearningVectorQuantization: # x = x + lr * (t - x ), lr = a * ( 1 - (epoch/max_epoch)) 
    def __init__(self, lr :float): # measuring similarity : || xi - wi || comp neural network
        self.lr = lr

    def fit(self, X):
        self.X = X

    def predict(self, X):
        pass 
    def __best_matching_units():
        pass

    @staticmethod
    def euclidean_distance(X1, X2):
        dist = 0.
        for i in range(X1):
            dist += np.sqrt(X1[i] - X2[i]) ** 2))
        return dist
