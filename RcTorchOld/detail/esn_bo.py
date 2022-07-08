# Based on the source code of GPyOpt/methods/modular_bayesian_optimization.py,
# which is licensed under the BSD 3-clause license

import time
import numpy as np
from GPyOpt.core.bo import BO
from GPyOpt.util.general import best_value
from GPyOpt.methods.modular_bayesian_optimization import ModularBayesianOptimization
from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence

__all__ = ['EchoStateBO']


class EchoStateBO(BO):
    """
    Modular Bayesian optimization, edited for stepwise application in Echo State Networks. 
    This class wraps the optimization loop around the different handlers.

    Adapted from ModularBayesianOptimization (Copyright (c) 2016, the GPyOpt Authors, BSD 3 license)

    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    """

    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, model_update_interval=1):

        self.initial_iter = True
        self.modular_optimization = True

        # Create optimization space
        super(EchoStateBO, self).__init__(
            model=model,
            space=space,
            objective=objective,
            acquisition=acquisition,
            evaluator=evaluator,
            X_init=X_init,
            Y_init=Y_init,
            cost=None,
            normalize_Y=False,  # Normalization done by RobustGPModel
            model_update_interval=model_update_interval)

    def run_target_optimization(self, target_score=0., max_iter=2048, max_time=np.inf, eps=1e-8, verbosity=True):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)
        :param max_iter: exploration horizon, or number of acquisitions.
            If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param verbosity: flag to print the optimization results after each iteration (default, True).
        :param report_file: filename of the file where the results of the optimization are saved (default, None).
        """
        # Save the options to print and save the results
        self.verbosity = verbosity

        # Avoid stacking: create full X and Y
        X_full = np.full((max_iter, self.X.shape[1]), fill_value=np.nan, dtype=np.float32)
        Y_full = np.full((max_iter, 1), fill_value=np.inf, dtype=np.float32)

        # Setting up stop conditions
        self.eps = eps
        self.max_iter = max_iter
        self.max_time = max_time
        self.target = target_score

        # Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_acquisitions = 0

        # Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            samples = self.X.shape[0]
            Y, _ = self.objective.evaluate(self.X)
            X_full[:samples, :] = self.X  # Confusingly, self.X and self.Y here are initial samples
            Y_full[:samples, :] = Y
            self.num_acquisitions += samples

        # Overwrite X and Y with full arrays to prevent stacking (slow)
        self.X = X_full
        self.Y = Y_full

        first_time = True

        # Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):

            # Update GP model
            self._update_model()

            # Inform user
            if first_time:
                if self.verbosity:
                    print("Model initialization done.", '\n')
                    print(self.model.model, '\n')
                    print(self.model.model.kern.lengthscale, '\n')
                first_time = False

            # Update and optimize acquisition and compute the exploration level in the next evaluation
            self.suggested_sample = self._compute_next_evaluations()

            # Augment X
            i = self.num_acquisitions
            sample_size = self.suggested_sample.shape[0]
            space_left = self.X.shape[0] - i
            allocation = np.clip(sample_size, 0, space_left)
            self.X[i:i + allocation, :] = self.suggested_sample[:allocation]

            # Evaluate *f* in X, augment Y
            Y_new = self.evaluate_objective()
            self.Y[i:i + allocation, :] = Y_new[:allocation]

            # Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            distance = self._distance_last_evaluations(sample_size)
            self.num_acquisitions += allocation

            # Target check
            if np.any(Y_new <= self.target):
                if self.verbosity:
                    pass
                print('Target reached at iteration', self.num_acquisitions)
                break

            # Check maximum iterations
            if self.num_acquisitions >= self.max_iter:
                if self.max_iter > 0 and self.verbosity:
                    print('Maximum iterations reached')
                break

            # Convergence
            if distance <= self.eps:
                if self.verbosity:
                    pass
                print('Converged at iteration', self.num_acquisitions)
                break

        # Stop messages and execution time
        self._compute_results()

        return self.num_acquisitions

    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        Y_new, _ = self.objective.evaluate(self.suggested_sample)
        return Y_new

    def _compute_next_evaluations(self):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        """
        return self.evaluator.compute_batch()

    def _distance_last_evaluations(self, sample_size):
        """
        Computes the distance between the last two evaluations.
        """
        i = self.num_acquisitions
        last = self.X[i, :]
        previous = self.X[i - sample_size, :]  # in Batch mode only the first sample is non-random
        distance = np.linalg.norm(last - previous, ord=1)  # Manhattan distance
        return distance

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        i = self.num_acquisitions
        if i % self.model_update_interval == 0:
            self.model.updateModel(self.X[:i], self.Y[:i])

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        i = self.num_acquisitions
        Y = self.Y[:i]
        self.Y_best = best_value(Y)
        self.x_opt = self.X[np.argmin(Y), :]
        self.fx_opt = np.min(Y)

    def plot_convergence(self, filename=None):
        """
        Makes twp plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        i = self.Y_best.shape[0]
        return plot_convergence(self.X[:i], self.Y_best, filename)

    def get_evaluations(self):
        i = self.num_acquisitions
        return self.X[:i], self.Y[:i]
