import numpy as np
np.random.seed(2022)

# Naive Bayes 
class NaiveBayes:
    def __init__(self):
        pass
    
    def predict(self, X):
        y_hat = [self.calcPosterior(x) for x in X]
        return np.argmax(np.array(y_hat), axis=-1)
    
    def fit(self, X, y):
        n_samples, n_feat = X.shape
        n_classes = len(np.unique(y))
        mu = np.zeros((n_classes, n_feat))
        sigma = np.zeros((n_classes, n_feat))
        prior = np.zeros(n_classes)
        
        for c in range(n_classes):
            X_c = X[y == c]
            mu[c, :] = np.mean(X_c, axis=0)
            sigma[c, :] = np.std(X_c, axis=0) ** 2
            prior[c] = X_c.shape[0] / n_samples
            
        self.mu = mu
        self.sigma = sigma
        self.prior = prior
        self.n_classes = n_classes

    def GaussianPDF(self, X, mu, sigma):
        gauss = 1 / np.sqrt(sigma * 2 * np.pi)
        prob = np.exp(-0.5 * ((X - mu) ** 2 / sigma))
        return gauss * prob
 
    def calcPosterior(self, X): # posterior
        posteriors = []
        for c in range(self.n_classes):
            mu = self.mu[c]
            sigma = self.sigma[c]
            prior = np.log(self.prior[c])
            posterior = np.log(self.GaussianPDF(X, mu, sigma)).sum()
            posterior = prior + posterior
            posteriors.append(posterior)
        return posteriors

# KNN
class KNearestNeighbor:
    def __init__(self, 
                 num_k : int = 5,
                 mode : str = "euclidean"):
        self.num_k = num_k
        self.mode = mode.lower()
        
    def fit(self, X, y):
        n_label = len(np.unique(y))
        self.n_samples, self.n_features = X.shape
        self.feature, self.label = X, y

    def predict(self, X_test):
        y_hat = []
        for i in X_test :
            distance = []
            for j in self.feature :
                if self.mode == "euclidean":
                    dist = KNearestNeighbor.euclideanDistance(i, j)
                elif self.mode == "manhattan":
                    dist = KNearestNeighbor.manhattanDistance(i, j)
                distance.append(dist)
            distance = np.array(distance)
            nearestNeighbor = np.argsort(distance)[:self.num_k]
            y_knn = KNearestNeighbor.findMaxVote(self.label, nearestNeighbor)
            y_hat.append(y_knn)
        return np.array(y_hat)

    @staticmethod
    def euclideanDistance(p, q):
        dist = 0
        for i in range(len(p)):
            dist += ((p[i] - q[i]))  ** 2
        return dist
        
    @staticmethod
    def manhattanDistance(p, q):
        dist = 0
        for i in range(len(p)):
            dist += np.abs((p[i] - q[i]))
        return dist
 
    @staticmethod
    def findMaxVote(y, neighbor):
        from collections import Counter
        vote = Counter(y[neighbor])
        return vote.most_common()[0][0]

# Logistic Regression
class LogisticRegression :
    def __init__(self, 
                 steps : int = 100, 
                 epsilon : float = 1e-6,
                 lr : float = 0.01,
                 threshold : float = 0.5,
                 use_bias : bool = True,
                 init : str = "normal", 
                 show_steps : bool = False):
        self.use_bias = use_bias
        self.steps = steps
        self.epsilon = epsilon
        self.init = init.lower()
        self.lr = lr
        self.show_steps = show_steps
        assert threshold < 1.0 and threshold > 0.0, f"Threshold has to be between 0 and 1 !"
        self.threshold = threshold
    
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        assert len(np.unique(y)) == 2, "Logistic Regression can only be used as binary classification. Use Softmax Regression instead."
        N, m = X.shape
        if self.init == "normal":
            w = np.random.normal(loc=0., scale=0.05, size=m)
            b = np.random.normal(loc=0., scale=0.05, size=1)
        elif self.init == "uniform":
            w = np.random.uniform(low=-0.05, high=0.05, size=m)
            b = np.random.uniform(low=-0.05, high=0.05, size=1)
        else :
            raise ValueError("Weights initializer is not valid. Use uniform or normal.")
        assert X.shape[0] == y.shape[0], f"Feature size {X.shape[0]} has not the same as label size {y.shape[0]}"            
        losses = []
        
        for _ in range(self.steps):
            if self.use_bias :
                y_prob = LogisticRegression.sigmoid(np.dot(X, w)) + b
                dw = (1 / m) * np.dot(X.T, (y_prob - y))
                db = (1 / m) * np.sum(y_prob - y)
                w = w - self.lr * dw
                b = b - self.lr * db
                loss = LogisticRegression.logloss(y, y_prob, self.epsilon)
                if self.show_steps:
                    print(f"Steps : {_ + 1} => Loss : {loss}")
                losses.append(loss)
            else : 
                y_prob = LogisticRegression.sigmoid(np.dot(X, w)) # feedforward
                dw = (1 / m) * np.dot(X.T, (y_prob - y))
                w = w - self.lr * dw
                loss = LogisticRegression.logloss(y, y_prob, self.epsilon)
                if self.show_steps :
                    print(f"Steps : {_ + 1} => Loss : {loss}")
                losses.append(loss)
        self.w, self.b = w, b
        
    def predict(self, X):
        assert X.shape[1] == len(self.w), "Different shape with fitted data !"
        if self.use_bias :
            z = LogisticRegression.sigmoid(np.dot(X, self.w)) + self.b
        else : 
            z = LogisticRegression.sigmoid(np.dot(X, self.w))
        return np.array([1 if i > self.threshold else 0 for i in z])
        
    @staticmethod
    def sigmoid(z):
        return 1 / ( 1 + np.exp(-z))
        
    @staticmethod
    def logloss(y_true, y_pred, epsilon):
        y_pred = np.clip(y_pred, a_min = epsilon, a_max = 1 - epsilon)
        notation1 = y_true * np.log(y_pred + epsilon)
        notation2 = ( 1 - y_true) * np.log(1 - y_pred + epsilon)
        notation = notation1 + notation2
        return - np.mean(notation)

# Softmax Regression
class SoftmaxRegression :
    def __init__(self, 
                 steps : int = 100, 
                 lr : float = 0.01,
                 use_bias : bool = True,
                 epsilon : float = 1e-7,
                 init : str = "normal", 
                 show_steps : bool = False):
        self.steps = steps
        self.lr = lr
        self.use_bias = use_bias
        self.init = init.lower()
        self.show_steps = show_steps
        
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        assert len(X) == len(y), f"Feature size {len(X)} has different size with label size len(y)"
        y_ohe = SoftmaxRegression.OneHot(y)
        losses = []
        N, m = X.shape
        if self.init == "normal":
            w = np.random.normal(0, 0.05, size=(m, y_ohe.shape[1]))
            b = np.random.normal(0, 0.05, size=(y_ohe.shape[1]))
        elif self.init == "uniform":
            w = np.random.uniform(low=-0.05, high=0.05, size=(m, y_ohe.shape[1]))
            b = np.random.uniform(low=-0.05, high=0.05, size=(y_ohe.shape[1]))
        else :
            raise ValueError("Weights initializer is not valid.. Use normal or uniform.")
        
        for _ in range(self.steps):
            if self.use_bias :
                z = SoftmaxRegression.softmax(np.dot(X, w)) + b
                dw = (1 / m) * np.dot(X.T, (z - y_ohe))
                db = (1 / m) * np.sum(z - y_ohe)
                w -= self.lr * dw
                b -= self.lr * db
                loss = SoftmaxRegression.categoryLogLoss(y, z)
                if self.show_steps :
                    print(f"Epochs : {_ + 1} => Loss : {loss}")
            else :
                z = SoftmaxRegression.softmax(np.dot(X, w))
                dw = (1 / m) * np.dot(X.T, (z - y_ohe))
                w -= self.lr * dw
                loss = SoftmaxRegression.categoryLogLoss(y, z)
                if self.show_steps :
                    print(f"Epochs: {_ + 1} => Loss : {loss}")
            losses.append(loss)
        self.m = m
        self.w, self.b = w, b
        
    def predict(self, X):
        X = np.array(X)
        assert X.shape[1] == self.m, f"{X.shape[1]} has not the same shape as fit !"
        if self.use_bias : 
            z = SoftmaxRegression.softmax(np.dot(X, self.w))
        else :
            z = SoftmaxRegression.softmax(np.dot(X, self.w))
        return np.argmax(z, axis=-1)
        
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    
    @staticmethod
    def OneHot(y):
        y_ohe = np.zeros((len(y), len(np.unique(y))))
        y_ohe[np.arange(len(y)), y] = 1
        return y_ohe
    
    @staticmethod
    def categoryLogLoss(y_true, y_pred):
        return - np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
