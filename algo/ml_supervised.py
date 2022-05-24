from collections import Counter
import numpy as np
np.random.seed(2022)

# Naive Bayes 
class GaussianNB:
    def __init__(self):
        pass

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

    def predict(self, X):
        y_hat = [self.__calcPosterior(x) for x in X]
        return np.argmax(np.array(y_hat), axis=-1)

    def __gaussianPDF(self, X, mu, sigma):
        gauss = 1 / np.sqrt(sigma * 2 * np.pi)
        prob = np.exp(-0.5 * ((X - mu) ** 2 / sigma))
        return gauss * prob
 
    def __calcPosterior(self, X): # posterior
        posteriors = []
        for c in range(self.n_classes):
            mu = self.mu[c]
            sigma = self.sigma[c]
            prior = np.log(self.prior[c])
            posterior = np.log(self.__gaussianPDF(X, mu, sigma)).sum()
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
        X, y = np.array(X), np.array(y)
        assert len(X) == len(y), f"Feature has {len(X)} while label has {len(y)}."
        n_label = len(np.unique(y))
        self.n_samples, self.n_features = X.shape
        self.feature, self.label = X, y

    def predict(self, X_test):
        assert X_test.shape[1] == self.feature.shape[1], f"Fitted feature has different size as predicted."
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
            y_knn = self.__findMaxVote(self.label, nearestNeighbor)
            y_hat.append(y_knn)
        return np.array(y_hat)

    @staticmethod
    def euclideanDistance(p, q):
        dist = 0
        for i in range(len(p)):
            dist += ((p[i] - q[i]))  ** 2
        return np.sqrt(dist)
        
    @staticmethod
    def manhattanDistance(p, q):
        dist = 0
        for i in range(len(p)):
            dist += np.abs((p[i] - q[i]))
        return dist
 
    def __findMaxVote(self, y, neighbor):
        vote = Counter(y[neighbor])
        return vote.most_common()[0][0]

# Logistic Regression
class LogisticRegression :
    def __init__(self, 
                 steps : int = 300, 
                 epsilon : float = 1e-6,
                 lr : float = 0.01,
                 threshold : float = 0.5,
                 use_bias : bool = True,
                 init : str = "normal"):
        self.use_bias = use_bias
        self.steps = steps
        self.epsilon = epsilon
        self.init = init.lower()
        self.lr = lr
        assert threshold < 1.0 and threshold > 0.0, f"Threshold has to be between 0 and 1 !"
        self.threshold = threshold
    
    def fit(self, X, y, batch_size=32, data_val : tuple = None):
        X, y = np.array(X), np.array(y)
        assert batch_size <= len(X), "Batch size can't be bigger than size of the data"
        assert len(np.unique(y)) == 2, "Logistic Regression can only be used as binary classification. Use Softmax Regression instead."
        if data_val is not None :
            assert len(data_val) == 2, "Validation data is only for feature and size !"
            X_val, y_val = data_val
            X_val, y_val = np.array(X_val), np.array(y_val)
            assert X_val.shape[1] == X.shape[1], f"Validation feature {data_val[0].shape[1]} has different column size while Train feature has {X.shape[1]} "
            assert len(np.unique(y_val)) == 2, "Logistic Regression can only be used as binary classification. Use Softmax Regression instead."
        N, m = X.shape
        if self.use_bias : 
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=m)
                b = np.random.normal(loc=0., scale=0.05, size=1)
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=m)
                b = np.random.uniform(low=-0.05, high=0.05, size=1)
            else :
                raise ValueError("Weights initializer is not valid. Use uniform or normal.")
        
        else :
            if self.init == "normal":
                w = np.random.normal(loc=0., scale=0.05, size=m)
            elif self.init == "uniform":
                w = np.random.uniform(low=-0.05, high=0.05, size=m)
            else :
                raise ValueError("Weights initializer is not valid. Use uniform or normal.")

        assert len(X) == len(y), f"Feature size {len(X)} has not the same as label size {len(y)}"            
        losses = []
        val_losses = []
        
        for _ in range(self.steps):
            perm = np.random.permutation(N)
            X_shuf, y_shuf = X[perm], y[perm]           
            
            for batch in range(0, N, batch_size):
                X_batch, y_batch = X_shuf[batch:batch+batch_size], y_shuf[batch:batch+batch_size]
                if self.use_bias :
                    y_prob = self.__sigmoid(np.matmul(X_batch, w) + b)
                    
                    dw, db = self.__gradientDescent(X_batch, y_batch, y_prob)
                    w, b = self.__update(w= w, dw= dw, b= b, db= db)
                    
                    loss = self.__logloss(y_batch, y_prob, self.epsilon)
                    losses.append(loss)
 
                    if data_val is not None:
                        val_prob = self.__sigmoid(np.matmul(X_val, w) + b)
                        val_loss = self.__logloss(y_val, val_prob, self.epsilon)
                        val_losses.append(val_loss)
                    
                else : 
                    y_prob = self.__sigmoid(np.matmul(X_batch, w)) # feedforward
                    
                    dw = self.__gradientDescent(X_batch, y_batch, y_prob)
                    w = self.__update(w= w, dw= dw)
                    
                    loss = self.__logloss(y_batch, y_prob, self.epsilon)
                    losses.append(loss)

                    if data_val is not None :
                        val_prob = self.__sigmoid(np.matmul(X_val, w))
                        val_loss = self.__logloss(y_val, val_prob, self.epsilon)
                        val_losses.append(val_loss)                   
 
        self.loss_hist, self.val_loss_hist = np.array(losses), np.array(val_losses)
        if self.use_bias :
            self.w, self.b = w, b
        else : 
            self.w = w
        
    def predict(self, X):
        assert X.shape[1] == len(self.w), f"Different shape {X.shape[1]} with fitted data {len(self.w)}!"
        if self.use_bias :
            z = self.__sigmoid(np.matmul(X, self.w) + self.b )
        else : 
            z = self.__sigmoid(np.matmul(X, self.w))
        return np.array([1 if i > self.threshold else 0 for i in z])
    
    def __gradientDescent(self, X, y_true, y_hat):
        if self.use_bias : 
            dw = (1 / len(X) * np.matmul(X.T, (y_hat - y_true))) # Calculate Gradient Descent for the weights
            db = (1 / len(X) * np.sum(y_hat - y_true)) # Calculate Gradient Descent for the bias
            return dw, db
        else :
            dw = (1 / len(X) * np.matmul(X.T, (y_hat - y_true))) # Calculate Gradient Descent for the weights
            return dw
    
    def __update(self, w, dw , b = None, db = None):
        if self.use_bias :
            w = w - self.lr * dw
            b = b - self.lr * db
            return w, b
        else : 
            w = w - self.lr * dw
            return w

    def __sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))
        
    def __logloss(self, y_true, y_pred, epsilon):
        y_pred = np.clip(y_pred, a_min = epsilon, a_max = 1 - epsilon)
        notation1 = y_true * np.log(y_pred + epsilon)
        notation2 = ( 1 - y_true) * np.log(1 - y_pred + epsilon)
        notation = notation1 + notation2
        return - np.mean(notation)

# Softmax Regression
class SoftmaxRegression :
    def __init__(self, 
                 steps : int = 300, 
                 lr : float = 0.01,
                 use_bias : bool = True,
                 init : str = "normal"):
        self.steps = steps
        self.lr = lr
        self.use_bias = use_bias
        self.init = init.lower()
        
    def fit(self, X, y, batch_size = 32, data_val : tuple = None):
        X, y = np.array(X), np.array(y)
        assert batch_size <= len(X), "Batch size can't be bigger than size of the data."
        assert len(X) == len(y), f"Feature size {len(X)} has different size with label size len(y)"
        if data_val is not None :
            assert len(data_val) == 2, "Validation data is only for feature and size !"
            X_val, y_val = data_val
            X_val, y_val = np.array(X_val), np.array(y_val)
            assert X_val.shape[1] == X.shape[1], f"Validation feature {data_val[0].shape[1]} has different column size while train feature has {X.shape[1]} "
        
        y_ohe = self.__OneHot(y)
        losses = []
        val_losses = []
        
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
            perm = np.random.permutation(N)
            X_perm, y_perm, y_perm_ohe = X[perm], y[perm], y_ohe[perm]
            
            for batch in range(0, N, batch_size):
                X_batch, y_batch, y_ohe_batch = X_perm[batch:batch+batch_size], y_perm[batch:batch+batch_size], y_perm_ohe[batch:batch+batch_size]
                if self.use_bias :
                    y_prob = self.__softmax(np.matmul(X_batch, w)) + b
                    dw, db = self.__gradientDescent(X_batch, y_ohe_batch, y_prob)
                    w, b = self.__update(w, dw, b, db)
                    loss = self.__categoryLogLoss(y_batch, y_prob)
                    losses.append(loss)
                    
                    if data_val is not None :
                        val_prob = self.__softmax(np.matmul(X_val, w) + b)
                        val_loss = self.__categoryLogLoss(y_val, val_prob)
                        val_losses.append(val_loss)
                     
                else :
                    y_prob = self.__softmax(np.matmul(X_batch, w))
                    dw = self.__gradientDescent(X_batch, y_ohe_batch, y_prob)
                    w = self.__update(w, dw)
                    loss = self.__categoryLogLoss(y_batch, y_prob)
                    losses.append(loss)
 
                    if data_val is not None :
                        val_prob = self.__softmax(np.matmul(X_val, w))
                        val_loss = self.__categoryLogLoss(y_val, val_prob)
                        val_losses.append(val_loss)

        self.w, self.b = w, b
        self.m = m
        self.loss_hist, self.val_loss_hist = np.array(losses), np.array(val_losses)
        
    def predict(self, X):
        X = np.array(X)
        assert X.shape[1] == self.m, f"{X.shape[1]} has not the same shape as fit !"
        if self.use_bias : 
            z = self.__softmax(np.matmul(X, self.w))
        else :
            z = self.__softmax(np.matmul(X, self.w))
        return np.argmax(z, axis=-1)
    
    def __gradientDescent(self, X, y_true, y_hat):
        if self.use_bias : 
            dw = (1 / len(X)) * np.matmul(X.T, (y_hat - y_true)) # Calculate gradient descent for the weights
            db =  (1 / len(X)) * np.sum(y_hat - y_true) # Calculate gradient descent for the bias
            return dw, db
        else : 
            dw = (1 / len(X)) * np.matmul(X.T, (y_hat - y_true))
            return dw

    def __update(self, w, dw, b = None, db = None):
        if self.use_bias:
            w = w - self.lr * dw
            b = b - self.lr * db
            return w, b
        else :
            w = w - self.lr * dw
            return w
        
    def __softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    
    def __OneHot(self, y):
        num_classes = len(np.unique(y))
        y_ohe = np.zeros((len(y), num_classes))
        y_ohe[np.arange(len(y)), y] = 1
        return y_ohe 
    
    def __categoryLogLoss(self, y_true, y_pred):
        return - np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

# Learning Vector Quantization
class LearningVectorQuantization: # x = x + lr * (t - x ), lr = a * ( 1 - (epoch/max_epoch)) 
    def __init__(self, lr :float = 0.01, epochs : int = 100): # measuring similarity : || xi - wi || comp neural network
        self.lr = lr
        self.epochs = epochs

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
            dist += ((X1[i] - X2[i]) ** 2)
        return np.sqrt(dist)

