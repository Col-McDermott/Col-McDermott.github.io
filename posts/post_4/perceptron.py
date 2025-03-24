# Col McDermott
# Implementation of Perceptron - Some code provided by Prof. Chodrow

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

class Perceptron(LinearModel):

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
    
    def grad(self, x, y):
        """
        Computes the perceptron update for a sampled point

        ARGUMENTS: 
            x, tch.Tensor: the sampled row vector of the feature matrix. x.size() == (1, p), 
            where p is the number of features.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        
        # Calculating the score for the sampled data point
        s_i = x @ self.w
        
        # The target corresponding to the sampled data point -- Might involve special indexing
       
        return (-1 * (((s_i * (2 * y) - 1) < 0) * 1)) * (y * x)

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        
        # Sampling a random data point
        i = tch.randint(X.size()[0], size = (1,))
        x_i = X[[i],:]
        y_i = y[i]

        # Computing the loss for the sampled point
        loss = self.model.loss(X, y)
        self.model.w -= self.model.grad(x_i, y_i)