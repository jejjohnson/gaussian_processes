import numpy as np
from scipy.linalg import solve_triangular

class GPUncertainty(object):
    """Gaussian Process Uncertainty Propagation

    This is a module that allows one to propagate the error given some
    input for a gaussian process.
    """
    def __init__(self, gp_model, x_error):
        self.gp_model = gp_model
        self.x_error = x_error


    def fit(self, X):

        # extract kernel parameters from previous GP Model

        # kernel parameters
        kernel_model = self.gp_model.kernel_
        self.signal_variance = kernel_model.get_params()['k1__k1__constant_value']
        self.length_scale = kernel_model.get_params()['k1__k2__length_scale']
        self.likelihood_variance = kernel_model.get_params()['k2__noise_level']

        # weights and data
        self.weights = self.gp_model.alpha_
        self.x_train = self.gp_model.X_train_

        # kernel matrices
        self.L = self.gp_model.L_
        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.K_inv = np.dot(L_inv, L_inv.T)

        # initialize parameters


        return self


    def predict(self, X, return_std=False):


        return None

    def propagate_error(self):


        return

    def get_covariance(self, u, x):

        diff = u - x
        D = np.dot(diff.T, self.)

        C_ux = self.signal_variance * \
            np.exp(-0.5)

        return C_ux

    def get_jacobian(self):

        return J_ux

    def get_hessian(self):

        return