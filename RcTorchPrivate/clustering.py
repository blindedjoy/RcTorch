import numpy as np
from .esn_cv import *
from .scr import *


__all__ = ['ClusteringBO']


class ClusteringBO(EchoStateNetworkCV):
    """Bayesian Optimization with an objective optimized for ESN Clustering (Maat & Gianniotis, 2017)

    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    readouts : array
        k-column matrix, where k is the number of clusters
    responsibilities : array
        matrix of shape (n, k) that contains membership probabilities for every series to every cluster

    """

    def __init__(self, bounds, assignment=None, target=0., eps=1e-6, initial_samples=100, max_iterations=300,
                 log_space=True, burn_in=30, seed=123, verbose=True, cv_samples=5, plot=True, **kwargs):

        # Initialize optimizer
        super().__init__(bounds, subsequence_length=-1, model=SimpleCycleReservoir, eps=eps,
                         initial_samples=initial_samples, max_iterations=max_iterations,
                         esn_burn_in=burn_in, random_seed=seed, verbose=verbose,
                         log_space=log_space, cv_samples=cv_samples, plot=plot, target_score=target,
                         **kwargs)

        # Save out weights for later
        self.assignment = assignment

        # Set objective accordingly
        if self.assignment is None:
            self.objective_sampler = self.k_folds_objective  # Multiple readouts
        else:
            self.objective_sampler = self.new_clustering_objective  # Weighted clustering objective: 1 readout

    def clustering_objective(self, parameters):
        # Get arguments
        arguments = self.construct_arguments(parameters)

        # Make simple sycle reservoir
        scr = self.model(**arguments)

        # How many series do we have
        n_series = self.x.shape[1]
        k_clusters = self.assignment.shape[1]

        # Simple check
        assert n_series == self.y.shape[1]

        # Placeholder
        scores = np.zeros((n_series, k_clusters), dtype=np.float32)

        # Generate error for every series
        for n in range(n_series):
            # Get series i
            x = self.x[:, n].reshape(-1, 1)
            y = self.y[:, n].reshape(-1, 1)

            # Compute score per cluster
            for k in range(k_clusters):
                scores[n, k] = scr.validation_score(y, x, folds=self.cv_samples, burn_in=self.esn_burn_in,
                                                    scoring_method=self.scoring_method)

        # Compute final scores
        final_score = np.mean(self.assignment * scores)

        # Inform user
        if self.verbose:
            print('Score:', final_score)

        return final_score.reshape(-1, 1)

    def new_clustering_objective(self, parameters):
        """WLS implementation for multiple series, with only 1 readout.

        Regression is weighted by class memberships.

        """
        # Get arguments
        arguments = self.construct_arguments(parameters)

        # How many series do we have
        n_series = self.x.shape[1]
        k_clusters = self.assignment.shape[1]

        # Simple check
        assert n_series == self.y.shape[1]

        # Placeholder
        scores = np.full((k_clusters, self.cv_samples), fill_value=np.nan, dtype=np.float32)

        # Train with multiple series
        for k in range(k_clusters):
            # Train simple sycle reservoir
            scr = self.model(**arguments)
            scores[k, :] = scr.train_validate_multiple(y=self.y, x=self.x, series_weights=self.assignment[:, k],
                                                       folds=self.cv_samples, burn_in=self.esn_burn_in,
                                                       scoring_method=self.scoring_method)
        # Compute final scores
        mean_score = scores.mean()

        # Inform user
        if self.verbose:
            print('Score:', mean_score)

        return mean_score.reshape(-1, 1)

    def k_folds_objective(self, parameters):
        """Does k-folds on the states for a set of parameters

        This method also deals with dispatching multiple series to the objective function if there are multiple,
        and aggregates the returned scores by averaging.

        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network

        Returns
        -------
        mean_score : 2-D array
            Column vector with mean score(s), as required by GPyOpt

        """
        # Get arguments
        arguments = self.construct_arguments(parameters)

        # Build network
        esn = self.model(**arguments)

        # Get all series
        y_all = self.y

        # Syntactic sugar
        n_series = y_all.shape[1]

        # Score placeholder
        scores = np.zeros(n_series, dtype=np.float32)

        for n in range(n_series):
            y = y_all[:, n].reshape(-1, 1)
            x = self.x[:, n].reshape(-1, 1)
            scores[n] = esn.validation_score(y, x, folds=self.cv_samples, burn_in=self.esn_burn_in,
                                             scoring_method=self.scoring_method)

        # Pass back as a column vector (as required by GPyOpt)
        mean_score = scores.mean()

        # Inform user
        if self.verbose:
            print('Score:', mean_score)

        # Return scores
        return mean_score.reshape(-1, 1)
