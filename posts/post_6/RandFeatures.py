# Col McDermott
# Sigmoidal Feature Map Implementation - Edited code from Prof. Chodrow

import torch as tch

# Logistic Sigmoid Utility Function
def sig(x): 
    return 1 / (1 + tch.exp(-x))

# Squaring Utility Function
def square(x): 
    return x ** 2

class RandomFeatures:
    """
    Random sigmoidal feature map. This feature map must be "fit" before use, like this: 

    phi = RandomFeatures(n_features = 10)
    phi.fit(X_train)
    X_train_phi = phi.transform(X_train)
    X_test_phi = phi.transform(X_test)

    model.fit(X_train_phi, y_train)
    model.score(X_test_phi, y_test)

    It is important to fit the feature map once on the training set and zero times on the test set. 
    """

    def __init__(self, n_features, activation = 0):
        self.n_features = n_features
        self.u = None
        self.b = None
        
        # Setting the activation function
        if activation == 0:
            self.activation = sig
        elif activation == 1:
            self.activation = square
        else:
            self.activation = activation

    # Initializing the transformation parameters
    def fit(self, X):
        self.u = tch.randn((X.size()[1], self.n_features), dtype = tch.float64)
        self.b = tch.rand((self.n_features), dtype = tch.float64) 

    # Applying a random feature transformation to the original data
    def transform(self, X):
        return self.activation((X @ self.u) + self.b)