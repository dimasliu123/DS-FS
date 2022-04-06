import numpy as np

def hamming(p, q): # P( A U B)
    dist = 0.
    for i in range(len(p)):
        dist += np.abs(p[i] - q[i])
    return dist

def euclidean(p, q): # f(x) = ∑ √( (pi - qi) ** 2)
    # note : Euclidean Distance is equivalent to L2 Normalization
    dist = 0
    for i in range(len(p)):
        dist += np.sqrt( (p[i] - q[i]) ** 2 )
    return dist

def manhattan(p, q):
    # note : Manhattan Distance is equivalent to L1 Normalization
    dist = 0
    for i in range(len(p)):
        dist += np.abs(p[i] - q[i])
    return dist

def minkowski(p, q, P): # f(x) = (| pi - qi | ** P) ** (1/P)
    # note : if P is 1, it's equivalent to Manhattan / Taxicab distance, if P is 2, it's equivalent to Euclidean Distance
    dist = 0
    for i in range(len(p)):
        dist += np.abs(p[i] - q[i]) ** P
    dist = dist ** (1 / P)
    return dist

def chebychev(p, q): # f(x) = max ( | xi - yi | )
    # note : Chebychev Distance is equivalent to L∞ norm
    p = np.array(p), np.array(q)
    dist = []
    for i in range(len(p)):
        dist.append(np.abs(p[i] - q[i]))
    return np.max(dist, axis=0)

def multi_dim_chebychev(p, q):
    p, q = np.array(p), np.array(q)
    dist = []
    for i in range(len(p)):
        for j in range(q.shape[1]):
            dist.append(np.abs(p[:, j][i] - q[:, j][i]))
    dist = np.array(dist).reshape(100, 5)
    return dist.max(axis=0)

def canberra(p, q): # f(x) = | pi - qi | / | pi + qi |
    dist = 0
    for i in range(len(p)):
        dist += np.abs(p[i] - q[i]) / np.abs(p[i] + q[i])
    return dist

def standard_euclidean(p, q): # f(x) = √(pi / std(p) / qi / std(q))
    # note : Taking the std of the vector to standardize the distance.
    dist = 0
    for i in range(len(p)):
        P = (p[i] / np.std(p, axis=0)) ** 2
        Q = (q[i] / np.std(q, axis=0)) ** 2
        dist += np.sqrt(P + Q)
    return dist

def chi_square_dist(p, q):
    dist = 0.
    for i in range(len(p)):
        dist += (p[i] - q[i]) ** 2 / (p[i] + q[i])
    return dist
