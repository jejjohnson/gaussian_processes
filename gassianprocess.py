import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='rbf', jitter=1e-10, random_state=None):
        self.kernel = kernel
        self.jitter = jitter
        self.random_state = random_state

        # Initialize heuristics
        self.signal_variance = 1.0
        self.length_scale = 1.0
        self.likelihood_variance = 0.01
    
    def fit(self, X, y):
        
        # Check inputs
        X, y = check_X_y(X, y)

        self.X_train_ = X
        self.y_train_ = y
        
        # Initial HyperParameters
        theta0 = np.array([self.signal_variance, 
                           self.length_scale, 
                           self.likelihood_variance])

        bounds = ((1e-7, 1e7), (1e-7, 1e7), (1e-7, 1e7))

        # Gradient Descent
        best_params = minimize(self.objective_function, x0=theta0, args=(),
                               method='L-BFGS-B', bounds=bounds, jac=True)

        # Get the best parameters
        self.signal_variance, self.length_scale, self.likelihood_variance = \
            self._get_hyperparams(best_params.x)
        
        self.best_neg_log_likelihood = best_params.fun 
        self.marginal_likelihood = np.exp(-best_params.fun)

        # Compute model
        K =

        return self
    
    # Objective Function
    def objective_function(self, theta, eval_gradient=True):

        if eval_gradient:
            lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True)
            return -lml, -grad
        else:
            return -self.log_marginal_likelihood
    
    def predict(self, X, y):
        
        return None

    def _get_hyperparams(self, theta):
        signal_variance = theta[0]
        likelihood_variance = theta[1]
        length_scale = theta[2:]
        return signal_variance, length_scale, likelihood_variance
        
    def negative_log_likelihood(self, ):

        return None


def main():
    pass

if __name__ == '__main__':
    main()