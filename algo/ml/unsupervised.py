import numpy as np
np.random.seed(2022)

class RBM:
    def __init__(self, n_visible : int, n_hidden : int = 256, lr : float = 0.1, num_k : int = 3):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.num_k = num_k
        self.W = np.random.uniform(low=0., high=1., size=(self.n_visible, self.n_hidden))
        self.vis = np.zeros(shape=(self.n_visible))
        self.hid = np.zeros(shape=(self.n_hidden))


    def fit(self, X, X_val = None, batch_size : int = 32, steps : int = 3000):
        X = np.array(X)
        N, m = X.shape
        loss = []

        for i in ( t:= tqdm(range(steps)) ):
            perm = np.random.permutation(N)
            X_perm = X[perm]

            for batch in range(0, N, batch_size):
                batch_err = []

                X_batch = X_perm[batch : batch + batch_size]

                # Positive Phase
                Z_pos = self.__P_HV(X_batch)
                Z_samp = self.__sample(Z_pos)
                posGrad = np.matmul(X_batch.T, Z_pos)

                # Negative Phase
                for k in range(self.num_k):
                    V = self.__P_VH(Z_pos)
                    V_gibbs = self.__sample(V)
                    Z_neg = self.__P_HV(V_gibbs)

                negGrad = np.matmul(V_gibbs.T, Z_neg)

                # Weights update
                self.W += self.lr * (posGrad - negGrad) * (1 / batch_size)
                self.hid += self.lr * np.sum(Z_pos, axis=0) - np.sum(Z_neg, axis=0) * (1 / batch_size)
                self.vis += self.lr * np.sum(V_gibbs, axis=0) - np.sum(X_batch, axis=0) * (1 / batch_size)

                loss_ = self.__energyLoss(X_batch, V_gibbs)
                t.set_description(f"Loss : {loss_}")
                batch_err.append(loss_)

            loss.append(np.mean(batch_err))

        self.loss = np.array(loss)

    def reconstruct(self, X):
        Z = self.__P_HV(X)
        Z_samp = self.__sample(Z)

        for k in range(self.num_k):
            V_gibbs = self.__P_VH(Z_samp)
            Z_neg = self.__P_HV(V_gibbs)
            Z_neg_samp = self.__sample(Z_neg)

        return V_gibbs

    def __P_HV(self, X):
        z = self.__sigmoid(np.matmul(X, self.W) + self.hid) # P( H | V )
        return z

    def __P_VH(self, z):
        H_z = self.__sigmoid(np.matmul(z, self.W.T) + self.vis) # P (V | H )
        return H_z

    def __sigmoid(self, z):
        return 1. / ( 1. + np.exp(-z) )

    def __sample(self, prob): # bernoulli if p == len(p), each pi that is sampled is independent.
        return bernoulli.rvs(prob)

    def __energyLoss(self, X, X_recon):
        err = np.sum(( X - X_recon ) ** 2)
        return err
