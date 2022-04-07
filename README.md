# Machine Learning Algorithm from scratch

## **Machine Learning** , **Feature Scaling**, From Scratch.

## Purpose : 
* The purpose of this project is to create a better intuition on the algorithm.
* Understand the intuition and the importance of scaling, and how each algorithm perform with different scaling.
* Understand the intuition of Distance measurement on real-world applications.

### Implementation example : 
``` Python
from algo.preprocess import Standardize
from algo.ml_supervised import KNearestNeighbor
sc = Standardize()
sc.calc(X_train)
sc_train = sc.scale(X_train)
sc_test = sc.scale(X_test)

knn = KNearestNeighbor()
knn.fit(sc_train, y_train)
knn_pred = knn.predict(sc_test)
```

## Implementation :

### Feature Preprocessing/Scaling :
* Min-Max Scaler
* Quantized Scaling
* Standardize/ Z-Score 

### Scaling Importance :
* Difference between each scaling from distribution.
![scaled](https://user-images.githubusercontent.com/86581543/162142204-aa1ab1c4-f5cc-4154-be08-773163cb0bd8.png)

### Supervised Learning :
* Naive Bayes
* K Nearest Neighbor
* Logistic Regression
* Softmax Regression

### Unsupervised Learning : 
* Will be updated.

### Deep Learning : 
* Will be updated.
