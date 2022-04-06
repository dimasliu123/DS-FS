import numpy as np

class MinMax:
    def __init__(self):
        pass

    def calc(self, X):
        dataMin, dataMax = np.min(X, axis=0), np.max(X, axis=0)
        self.dataMin, self.dataMax = dataMin, dataMax

    def scale(self, X): # f(x) = ( X - Xmin) / ( Xmax - Xmin) 
        X_scaled = (X - self.dataMin) / (self.dataMax - self.dataMin)
        X_scaled[X_scaled == np.nan] = 0.
        return X_scaled

class Standardize :
    def __init__(self, with_mu =True, with_std =True):
        self.with_mu = True
        self.with_std = True

    def calc(self, X): # f(x)  = ( X - mu) / sigma => mu : mean, sigma ; std
        if self.with_mu == True:
            X_mu = np.mean(X, axis=0)
        else :
            X_mu = np.zeros(X.shape[1])
        if self.with_std == True:
            X_std = np.std(X, axis=0)
        else :
            X_std = np.full((X.shape[1]), 1)
        self.X_std, self.X_mu = X_std, X_mu

    def scale(self, X):
        Z_Score = (X - self.X_mu) / self.X_std
        Z_Score[Z_Score == np.nan] = 0.
        return Z_Score

class QuantizedScaling:
    def __init__(self, scaling_range=(0.2, 0.8)):
        self.scaling_range = scaling_range
        assert self.scaling_range[0] > 0.0 and self.scaling_range[1] < 1.0, "Scaling range has to be between 0. and 1."

    def calc(self, X): # f(x) = ( X - median) / ( Xq3 - Xq1 )
        q1 = np.quantile(X, self.scaling_range[0], axis=0)
        med = np.median(X, axis=0)
        q3 = np.quantile(X, self.scaling_range[1], axis=0)
        self.q1, self.med, self.q3 = q1, med, q3

    def scale(self, X):
        X_scaled = ( X - self.med ) / (self.q3 - self.q1)
        X_scaled[X_scaled == np.nan] = 0.
        return X_scaled

class OneHot:
    def __init__(self):
        pass
    
    def calc(self, X):
        self.numCategory = len(np.unique(X))
    
    def scale(self, X):
        zeros = np.zeros((len(X), self.numCategory))
        zeros[np.arange(len(X)), X] = 1
        return zeros
