

class GaussianProcessAutograd(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='rbf', jitter=1e-09, random_state=None):
        self.kernel = kernel
        self.jitter = jitter
        self.random_state = random_state

    def fit(self, X, y):
        
        # standardize the data
        self.X = X
        self.Y = y
        
        # initialize parameters
        noise_scale = 0.01
        length_scale = 10.0
        theta0 = np.array([noise_scale, length_scale])
        bounds = ((1e-10, 1e10), (1e-10, 1e10))
        
        # define objective function: negative log marginal likelihood
        objective = lambda theta0: -self.log_marginal_likelihood(theta0)
        
        # minimize negative log marginal likelihood
        cov_params = minimize(value_and_grad(objective), theta0, jac=True, args=(), method='L-BFGS-B', bounds=bounds)
        
        # get params
        self.noise_scale, self.length_scale = self._unpack_kernel_params(cov_params.x)
        
        # calculate the weights
        K = self.K(self.X, length_scale=self.length_scale)
        self.L = np.linalg.cholesky(K + noise_scale * np.eye(K.shape[0]))
        weights =  np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y))
        self.weights_ = weights
        
        return self
    
    def K(self, X, Y=None, length_scale=1.0, scale=1.0):
        
        scale_term = - 0.5 / np.power(length_scale, 2)

        if Y is None:

            dists = pdist(X, metric='sqeuclidean')

            K = np.exp(scale_term * dists)

            K = squareform(K)

            np.fill_diagonal(K, 1)

        else:

            dists = cdist(X, Y, metric='sqeuclidean')

            K = np.exp(scale_term  * dists) 


        return K
    
    def predict(self, X, return_std=None):
        
        K = self.K(X, self.X, length_scale=self.length_scale)
        
        predictions = np.dot(K, self.weights_)
        if not return_std:
            return predictions
        else:            
            K_test = rbf_covariance(X, length_scale=self.length_scale)
            v = np.linalg.solve(self.L, K.T)
            std_dev = np.sqrt(self.noise_scale + np.diag(K_test - np.dot(v.T, v)))
            return predictions, std_dev
    
    def pred_grad(self, X):
        
        mu = lambda x: self.predict(x, return_std=False)
        auto_grad = autograd.grad(mu)
        
        return auto_grad(X)
    
    def _unpack_kernel_params(self, params):
        return params[0], params[1:]
    
    def log_marginal_likelihood(self, params):
        
        x_train = self.X
        y_train = self.Y
        
        # unpack the parameters
        noise_scale, length_scale = self._unpack_kernel_params(params)
        ktrain = self.K(x_train, length_scale=length_scale) 
        white_kern = noise_scale * np.eye(len(y_train))
        print(ktrain.shape, white_kern.shape)
        K = ktrain + white_kern
#         # calculate the covariance matrix
#         K = self.K(x_train, length_scale=length_scale)
#         K_chol = K + noise_scale * np.eye(K.shape[0])
# #         K += self.jitter * np.eye(K.shape[0])
        
#         # Solve the cholesky
#         print(K.shape)
#         try:
#             self.L = np.linalg.cholesky(K_chol)
            
#         except np.linalg.LinAlgError:
#             return -np.inf
                
#         if y_train.ndim == 1:
#             y_train = y_train[:, np.newaxis]
        
#         # get the weights
#         alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_train))
        
#         # compute log-likelihood
#         log_likelihood_dims = -0.5 * np.einsum('ik,ik->k', y_train, alpha)
#         log_likelihood_dims -= np.log(np.diag(self.L)).sum()
#         log_likelihood_dims -= (K.shape[0] / 2 ) * np.log(2 * np.pi)
        
#         log_likelihood = log_likelihood_dims.sum(-1)
        tmp = mvn.logpdf(y_train, 0.0, Kernel)
        print(tmp)
        return tmp
