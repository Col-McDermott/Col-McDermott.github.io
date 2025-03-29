# Col McDermott
# Implementation of Minibatch Perceptron - Some code provided by Prof. Chodrow

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
        if self.w is None: 
            self.w = tch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return X @ self.w

    def predict(self, X):
        """
        Computes the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, tch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        
        return (self.score(X) > 0.5).float()

class MBPerceptron(LinearModel):

    def loss(self, X, y):
        """
        Computes the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """

        return tch.mean(((self.score(X) * ((2 * y) - 1)) <= 0).float())
    
    def grad(self, X, y, lr):
        """
        Computes the perceptron update for a sampled point

        ARGUMENTS: 
            X, tch.Tensor: a submatrix of size k X p of the feature matrix. X.size() == (k, p), 
            where p is the number of features.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            lr: learning rate parameter
        """
        
        # Calculating the score for each data point in the submatrix
        s_k = self.score(X)

        # Calculating the misclassification rate of the submatrix
        mc = ((s_k * ((2 * y) - 1) < 0)) * 1
        
        # Calculating the submatrix row updates
        ud = X * (((2 * y) - 1).reshape(-1, 1))
    
        # Determining if each row of the submatrix should be updated according its classification correctness
        ur = mc.reshape(-1, 1) * ud

        # Computing the average of the row updates
        ur_avg = ur.mean(0, True)
    
        # Calculating the of the updates from the data points in the submatrix scaled by the learning rate
        grad = -1 * ((lr * ur_avg)[0])

        return grad

class MBPerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, lr):
        """
        Compute one step of the perceptron update using a row of 
        the feature matrix X and corresponding target vector value of y.

        lr: learning rate parameter to pass to grad() method 
        """

        # Computing the loss for the sampled point and conducting a the next perceptron step is necessary
        if self.model.loss(X, y) > 0:
            self.model.w -= self.model.grad(X, y, lr)
            return 1
        return 0