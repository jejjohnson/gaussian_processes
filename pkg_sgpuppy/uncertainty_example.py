import numpy as np
from skgpuppy.GaussianProcess import GaussianProcess
from skgpuppy.Covariance import GaussianCovariance

# Preparing some parameters (just to create the example data)
x = np.array([[x1,x2] for x1 in xrange(10) for x2 in xrange(10)]) # 2d sim input (no need to be a neat grid in practice)
w = np.array([0.04,0.04])   # GP bandwidth parameter
v = 2                       # GP variance parameter
vt = 0.01                   # GP variance of the error epsilon

# Preparing the parameter vector
theta = np.zeros(2+len(w))
theta[0] = np.log(v)  # We actually use the log of the parameters as it is easier to optimize (no > 0 constraint etc.)
theta[1] = np.log(vt)
theta[2:2+len(w)] = np.log(w)

# Simulating simulation data by drawing data from a random Gaussian process
t = GaussianProcess.get_realisation(x, GaussianCovariance(),theta)

# The regression step is pretty easy:
# Input data x (list of input vectors)
# Corresponding simulation output t (just a list of floats of the same length as x)
# Covariance function of your choice (only GaussianCovariance can be used for uncertainty propagation at the moment)
gp_est = GaussianProcess(x, t,GaussianCovariance())

# Getting some values from the regression GP for plotting
x_new = np.array([[x1/2.0,x2/2.0] for x1 in xrange(20) for x2 in xrange(20)])
means, variances = gp_est.estimate_many(x_new)

# Plotting the output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x.T[0],x.T[1],t, cmap=cm.autumn, linewidth=0.2)
ax.plot_trisurf(x_new.T[0],x_new.T[1],means, cmap=cm.winter, linewidth=0.2)
plt.show()


#####################################
# Uncertainty Propagation
#####################################

# Continuing the regression example

from skgpuppy.UncertaintyPropagation import UncertaintyPropagationApprox

# The uncertainty to be propagated
mean = np.array([5.0,5.0]) # The mean of a normal distribution
Sigma = np.diag([0.01,0.01]) # The covariance matrix (must be diagonal because of lazy programming)

# Using the gp_est from the regression example
up = UncertaintyPropagationApprox(gp_est)

# The propagation step
out_mean, out_variance = up.propagate_GA(mean,Sigma)

print out_mean, out_variance