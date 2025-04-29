# Col McDermott
# Implementation of logistic regression with advanced optimization methods - Some code provided by Prof. Chodrow

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
        Computes the empirical risk L(w) using the logistic loss function
        
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
        er = tch.mean((-y * tch.log2(self.sig(s) + 1e-10)) - ((1 - y) * tch.log2(1 - self.sig(s) + 1e-10))).item() # The value 1e-10 is added to each term inside a log(x) operation to avoid nan computation

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
    
    def hessian(self, X):
        """
        Computes the Hessian of the empirical risk function (L(w)):

        h_{i, j}(w) = sum_{k=1}^{n}(x_{k, i} * x_{k, j} * sig(s_k) * (1 - sig(s_k)))

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.
        """

        # Computing the scores
        s = self.score(X)

        # Computing a diagonal matrix where each non-zero entry D[k, k] = sig(s_k) * (1 - sig(s_k)) 
        D = tch.diag(self.sig(s) * (1 - self.sig(s))) 
        
        # Computing the Hessian matrix using the feature matrix X and the diagonal matrix D
        H = X.t() @ (D @ X)

        # Normalizing the hessian by n to ensure numerical stability with respect to the gradient
        H = (H / X.size(0)).float() 
        
        # Adding a small regularization value to each entry to avoid a value of 0.0 - letting H be invertible
        H = (H + (1e-10 * tch.eye(H.size(0)))).float()

        return H

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
        # Initializing a previous weights vector if it currently does not exist
        if (self.prev_w == None):
            self.prev_w = self.lr.w

        # Computing the gradient descent update 
        ud = self.lr.w - (alpha * self.lr.grad(X, y)) + (beta * (self.lr.w - self.prev_w))
        
        # Updating the weights vectors
        self.prev_w = self.lr.w
        self.lr.w = ud

        return ud
    
    def optimizeSGD(self, X, y, batch_size, alpha, beta):
        """
        Performs standard stochastic gradient descent with momentum: 
        
        Performs this update on each data at each step:
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
        
        # Data loader for batching the data
        dl = tch.utils.data.DataLoader(
            tch.utils.data.TensorDataset(X, y),
            batch_size = batch_size,
            shuffle = True
        )

        # Initializing the weights vector is necessary
        if (self.lr.w == None):
            self.lr.w = tch.rand((X.size()[1]))
        
        # Initializing the previous weights vector if it currently does not exist
        if (self.prev_w == None):
            self.prev_w = self.lr.w

        # Iterating through each data batch
        for X_k, y_k in dl:
            
            # Computing the gradient descent update 
            ud = self.lr.w - (alpha * self.lr.grad(X_k, y_k)) + (beta * (self.lr.w - self.prev_w))
        
            # Updating the weights vectors
            self.prev_w = self.lr.w
            self.lr.w = ud

    
class NewtonOptimizer:

    # Instance variable for a logistic regression model
    def __init__(self, lr):
        self.lr = lr

    def step(self, X, y, alpha):
        """
        Performs a step of Newton's method: 
        
        w_{k+1} <- w_k - alpha * H(X)^{-1} * grad(X, y)

        - w_k is the current weights vector
        - H(X)^{-1} is the inverse of the Hessian of the empirical risk function (L(w))
        - grad(X, y) is the gradient of the empirical risk function (L(w))
        - alpha is the learning rate for the gradient descent process

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            alpha, float: the learning rate for the gradient descent process
        """

        # Computing the inverse of the Hessian matrix - Note that the Moore-Penrose pseudoinverse is used in case the hessian is singular
        H_inv = tch.linalg.pinv(self.lr.hessian(X))

        # Computing the gradient of the loss function
        grad = self.lr.grad(X, y)

        # Computing the Newton update
        self.lr.w = self.lr.w - (alpha * (H_inv @ grad))

    def optimize(self, X, y, alpha, tol):
        """
        A method to run Newton's method until the model's empirical loss value reaches the desired tolerance

        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            alpha, float: the learning rate for the gradient descent process

            tol, float: the desired loss-value tolerance
        """
        
        # Optimization loop
        while (self.lr.loss(X, y) > tol):

            # If not converged, perform an optimization step
            self.step(X, y, alpha)


class AdamOptimizer:

    # Instance variable for a logistic regression model
    def __init__(self, lr):
        self.lr = lr

    def optimizeEpoch(self, X, y, batch_size, alpha, beta_1, beta_2, w_0 = None):
        """
        Performs the Adam Algorithm, updates the weights vector for each batch of data using: 
        
        w_{k+1} <- w_k - alpha * m^_t / ((sqrt(v^_t) + epsilon)
        
        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            batch_size, int: Parameter to select a subset of the feature matrix. X_subset.size() == (batch_size, p), 
            
            alpha, float: the learning rate for the gradient descent process

            beta_1, float: the first moment (mean of gradient) decay rate

            beta_2, float: the second raw moment (un-centered variance of gradient) decay rate

            w_0, tch.Tensor: initial guess for the weights vector w. w.size() == (p, 1)
        """

        # Data loader for batching the data
        dl = tch.utils.data.DataLoader(
            tch.utils.data.TensorDataset(X, y),
            batch_size = batch_size,
            shuffle = True
        )

        # Initializing the weights vector is necessary
        if (w_0 == None):
            self.lr.w = tch.rand((X.size()[1]))
        else:
            self.lr.w = w_0

        # Initializing 1st, 2nd moment vectors
        m = tch.zeros(self.lr.w.size())
        v = tch.zeros(self.lr.w.size())

        # Iterating through each batch in the data
        t = 0 # Initialize timestep
        for X_k, y_k in dl:
            
            # Update timestep
            t += 1
            
            # Get the current gradient
            g = self.lr.grad(X_k, y_k)
            
            # Update the 1st, 2nd moment vectors
            m = (beta_1 * m) + ((1 - beta_1) * g)
            v = (beta_2 * v) + ((1 - beta_2) * (g ** 2))

            # Update the bias correction terms
            m_hat = m / (1 - (beta_1 ** t))
            v_hat = v / (1 - (beta_2 ** t))

            # Update the weights vector w
            self.lr.w = self.lr.w - (alpha * (m_hat / (tch.sqrt(v_hat) + 1e-8))) # Epsilon term set to 1e-8 as recommended by the alg. designers

    def optimize(self, X, y, tol, batch_size, alpha, beta_1, beta_2, w_0 = None):
        """
        Performs the Adam Algorithm until the loss value reaches a desired tolerance.
        Updates the weights vector for each batch of data using: 
        
        w_{k+1} <- w_k - alpha * m^_t / ((sqrt(v^_t) + epsilon)
        
        ARGUMENTS:
            X, tch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, tch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            batch_size, int: Parameter to select a subset of the feature matrix. X_subset.size() == (batch_size, p), 
            
            alpha, float: the learning rate for the gradient descent process

            beta_1, float: the first moment (mean of gradient) decay rate

            beta_2, float: the second raw moment (un-centered variance of gradient) decay rate

            w_0, tch.Tensor: initial guess for the weights vector w. w.size() == (p, 1)

            tol, float: the desired tolerance
        """

        # Optimization loop
        while (self.lr.loss(X, y) > tol):
            self.optimizeEpoch(X, y, batch_size, alpha, beta_1, beta_2, w_0 = self.lr.w)