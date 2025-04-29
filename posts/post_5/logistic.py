# Col McDermott
# Implementation of logistic regression - Some code provided by Prof. Chodrow

import torch as tch

class LinearModel:

    def __init__(self):
        self.w = None

    def score(self, X):
        """
        Computes the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s tch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w == None:
            self.w = tch.rand((X.size()[1]))
        
        return X @ self.w

    def predict(self, X):
        """
        Computes the predictions for each data point in the feature matrix X. 
        The prediction for the ith data point is either 0 (if the its score is less than 0) or 1 (otherwise). 

        ARGUMENTS: 
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, tch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        
        return (self.score(X) > 0.5).float()


class LogisticRegression(LinearModel):
    
    # Logistic sigmoid function
    def sig(self, x):
        epsilon = 0 # Additive factor to avoid NaN results when computing loss
        sig = (1 / (1 + tch.exp(-x)))

        return sig + epsilon

    def loss(self, X, y):
        """
        Computes the empirical risk L(w) using the logistic loss function:
        
        L(W) = (1 / n) * sum((-y_i * log(sig(s_i))) - (1 - y_i) * log(1 - sig(s_i)))

        - s_i is the score of ith data point

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        # Computing the scores
        s = self.score(X)

        # Computing the empirical risk given the current weights vector
        er = tch.mean((-y * tch.log2(self.sig(s))) - ((1 - y) * tch.log2(1 - self.sig(s)))).item()

        return er
    
    def grad(self, X, y):
        """
        Computes the gradient of the empirical risk function (L(w)):

        grad(L(w)) = (1 / n) * sum((sig(s_i) - y_i) * x_i)

        - s_i is the score of ith data point

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        # Computing the scores
        s = self.score(X)
        
        # Computing the gradient of the empirical risk function given the current weights vector 
        grad = ((self.sig(s) - y)[:, None] * X).mean(0, True).squeeze()

        return grad

class GradientDescentOptimizer:

    # Instance variables for a logistic regression model and the previous weights vector 
    def __init__(self, lr):
        self.lr = lr
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        """
        Performs a step of gradient descent with momentum: 
        
        w_{k+1} <- w_k - alpha * grad(w_k) + beta * (w_k - w_{k-1})

        - w_k is the current weights vector
        - w_{k-1} is the previous weights vector

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            alpha, float: the learning rate for the gradient descent process

            beta, float: the momentum factor     
        """
        # Initializing a weights vector if it currently does not exist
        if (self.prev_w == None):
            self.prev_w = self.lr.w

        # Computing the gradient descent update 
        ud = self.lr.w - (alpha * self.lr.grad(X, y)) + (beta * (self.lr.w - self.prev_w))
        
        # Updating the weights vectors
        self.prev_w = self.lr.w
        self.lr.w = ud

        return ud