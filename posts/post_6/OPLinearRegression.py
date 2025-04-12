# Col McDermott
# Overparameterized Linear Regression Model Implementation - Some code provided by Prof. Chodrow

import torch as tch
import numpy as np

class LinearModel:

    # Storing the weights vector w
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

class LinearRegression(LinearModel):

    def predict(self, X):
        """
        Computes the predictions for each data point in the feature matrix X. 
        The prediction for the ith data point is numeric value. 

        ARGUMENTS: 
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, tch.Tensor: vector predictions in R. y_hat.size() = (n,)
        """
        
        y_hat = self.score(X)

        return y_hat
    
    def loss(self, X, y):
        """
        Computes the mean-squared-error (MSE) loss between the scores s and the targets y
        
        MSE = (1 / n) * sum((s_i - y_i)^2)

        - s_i is the score of ith data point

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,).
        
            RETURNS:
            mse, the MSE value of the linear regression model with its current weights vector w
        """

        # Computing the scores
        s = self.score(X)

        # Computing the MSE
        mse = tch.mean((s - y) ** 2).item()

        return mse
    
class OPLinearRegressionOptimizer:

    # Constructed with an implicit linear regression model
    def __init__(self, lr):
        self.lr = lr

    def fit(self, X, y):
        """
        Solves for the optimal weights vector w using the Moore-Penrose pseudoinverse of the feature matrix X

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
                where n is the number of data points and p is the 
                number of features. This implementation always assumes 
                that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,).
        """

        # Computing the Moore-Penrose pseudoinverse of X
        X_pinv = tch.linalg.pinv(X)
        
        # Setting the weights vector w of the implicit linear regression model to the optimal weights vector
        self.lr.w = X_pinv @ y
