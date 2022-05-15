import numpy as np
import matplotlib.pyplot as plt
from algo.ml_supervised import *
from algo.preprocess import *

plt.style.use("seaborn-darkgrid")

np.random.seed(42)

X_train = np.random.normal(loc=0., scale=0.05, size=(1000, 5))
y_train = np.random.randint(0, 2, size=(1000))

X_val = np.random.normal(loc=0., scale=0.05, size=(200, 5))
y_val = np.random.randint(0, 2, size=(200))

lr = LogisticRegression()
lr.fit(X_train, y_train)

plt.plot(lr.loss_hist, color='b')
