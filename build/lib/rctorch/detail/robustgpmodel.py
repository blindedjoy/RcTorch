# Based on the source code of GPyOpt/models/gpmodel.py,
# which is licensed under the BSD 3-clause license

import numpy as np
import GPyOpt
import GPy
import copy
import contextlib
import sys
import traceback
from scipy.special import gammaln, digamma
from paramz.domains import _REAL, _POSITIVE


__all__ = ['RobustGPModel']


class RobustGPModel(GPyOpt.models.GPModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    Adapted from GPModel (Copyright (c) 2016, the GPyOpt Authors, BSD 3 license)

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available.
    :param normalize_Y: normalization of the outputs to the interval [0,1] (default, True).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """
    analytical_gradient_prediction = True

    def __init__(self, noise_var=None, log_space=False, exact_feval=False, normalize_Y=True,
                 max_iters=1000, verbose=True, **kwargs):
        super().__init__()
        self.noise_var = noise_var
        self.log_space = log_space
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None

    def _preprocess_data(self, X, Y):
        # Remove non-finite values
        finite_mask = np.isfinite(Y.ravel())  # TODO: Detect outliers that are not infinite
        infinite_indices = np.nonzero(~finite_mask)[0]

        # Replace with mean
        if not self.model is None and infinite_indices.shape[0] > 0:
            X_inf = X[infinite_indices]
            means, stds = self.model.predict(X_inf)
            Y[infinite_indices] = means + 1e-8 * stds
        elif infinite_indices.shape[0] > 0:
            Y = Y[finite_mask]
            X = X[finite_mask]

        # Log if wanted
        if self.log_space:
            Y = np.log(Y + 1e-8)  # Y will be negative when working with (N)MSE

        # Normalize
        if self.normalize_Y:
            Y -= Y.mean()
            std = Y.std()
            Y /= std if std > 0. else 1.

        return X, Y

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        with Suppressor():
            # Kernel and priors
            self.input_dim = X.shape[1]
            kernel = GPy.kern.Matern52(self.input_dim, ARD=True)
            prior = FixedInverseGamma(.001, .001)
            kernel.lengthscale.set_prior(prior)
            kernel.variance.set_prior(prior)

            # Model
            noise_var = 1. if self.noise_var is None else self.noise_var
            self.model = GPy.models.GPRegression(X, Y, kernel=kernel, noise_var=noise_var)
            self.model.likelihood.variance.set_prior(prior)

            # Evaluation constriant
            if self.exact_feval or noise_var < 1e-6:
                self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
            else:
                self.model.Gaussian_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all):
            """
            Updates the model with new observations.
            """
            X, Y = self._preprocess_data(X_all, Y_all)

            if self.model is None:
                self._create_model(X, Y)
            else:
                self.model.set_XY(X, Y)

            # Update model
            self.model.optimize(optimizer='lbfgs', messages=False, max_iters=self.max_iters,
                                ipython_notebook=True, clear_after_finish=True)

    def predict(self, X):
            """
            Predictions with the model. Returns posterior means and standard deviations at X.
            Note that this is different in GPy where the variances are given.
            """
            if X.ndim == 1:
                X = X[None, :]
            m, v = self.model.predict(X)
            v = np.clip(v, 1e-10, np.inf)
            return m, np.sqrt(v)

    def get_fmin(self):
            """
            Returns the location where the posterior mean is takes its minimal value.
            """
            return self.model.predict(self.model.X)[0].min()

    def predict_withGradients(self, X):
            """
            Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
            """
            if X.ndim == 1:
                X = X[None, :]
            m, v = self.model.predict(X)
            v = np.clip(v, 1e-10, np.inf)
            dmdx, dvdx = self.model.predictive_gradients(X)
            dmdx = dmdx[:, :, 0]
            dsdx = dvdx / (2 * np.sqrt(v))
            return m, np.sqrt(v), dmdx, dsdx

    def copy(self):
            """
            Makes a safe copy of the model.
            """
            return copy.deepcopy(self)

    def get_model_parameters(self):
            """
            Returns a 2D numpy array with the parameters of the model
            """
            return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
            """
            Returns a list with the names of the parameters of the model
            """
            return self.model.parameter_names()


# Added to circumvent a GPy bug
class FixedInverseGamma(GPy.priors.Gamma):
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.
    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)
    .. Note:: Bishop 2006 notation is used throughout the code
    """
    domain = _POSITIVE

    def __init__(self, a, b):
        super().__init__(a, b)
        self._a = float(a)
        self._b = float(b)
        self.constant = -gammaln(self._a) + self._a * np.log(self._b)

    def __str__(self):
        return "iGa({:.2g}, {:.2g})".format(self._a, self._b)

    def lnpdf(self, x):
        return self.constant - (self._a + 1) * np.log(x) - self._b / x

    def lnpdf_grad(self, x):
        return -(self._a + 1.) / x + self._b / x ** 2

    def rvs(self, n):
        return 1. / np.random.gamma(scale=1. / self._b, shape=self._a, size=n)


class Suppressor(object):
    """
    Adapted from: https://stackoverflow.com/questions/2828953/ \
        silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto/40054132#40054132

    """

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type_, value, trace):
        sys.stdout = self.stdout
        if type_ is not None:
            # Do normal exception handling
            traceback.print_tb(trace)

    def write(self, x):
        pass
