# Derivatives and Gaussian Processes

## Notation

* $K$
    * kernel function (training points)
    * $NxN$
* $Ky=\alpha$
    * weights of the matrix
    * $Nx1$
* $k(x_*, X)=k_*$
    * Test-Train Kernel
    * $1xN$

## Key Functions

**Posterior Mean** of a GP:

$$\bar{f_*} = k_* K^{-1}y = k_* \alpha$$

### Distribution for 1st Order Derivatives of Posterior Functions

**Derivative**

$$\frac{\partial k(x_*, x_i)}{\partial x_*} = - \lambda (x_* - x_i) k(x_*, x_i)$$

### Taylor Expansion

### Expected Squared Derivative

$$V(X) = \mathcal{E}(X^2) - \mathcal{E}(X)^2$$
