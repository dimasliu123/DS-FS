import numpy as np
np.random.seed(2022)

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
