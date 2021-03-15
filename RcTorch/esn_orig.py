import numpy as np
import scipy.stats
import scipy.linalg
import copy
import json

__all__ = ['EchoStateNetwork']


class EchoStateNetwork:
    """Class with all functionality to train Echo State Nets.

    Builds and echo state network with the specified parameters.
    In training, testing and predicting, x is a matrix consisting of column-wise time series features.
    Y is a zero-dimensional target vector.

    Parameters
    ----------
    n_nodes : int
        Number of nodes that together make up the reservoir
    input_scaling : float
        The scaling of input values into the network
    feedback_scaling : float
        The scaling of feedback values back into the reservoir
    spectral_radius : float
        Sets the magnitude of the largest eigenvalue of the transition matrix (weight matrix)
    leaking_rate : float
        Specifies how much of the state update 'leaks' into the new state
    connectivity : float
        The probability that two nodes will be connected
    regularization : float
        The L2-regularization parameter used in Ridge regression for model inference
    feedback : bool
        Sets feedback of the last value back into the network on or off
    random_seed : int
        Seed used to initialize RandomState in reservoir generation and weight initialization

    Methods
    -------
    train(y, x=None, burn_in=100)
        Train an Echo State Network
    test(y, x=None, y_start=None, scoring_method='mse', alpha=1.)
        Tests and scores against known output
    predict(n_steps, x=None, y_start=None)
        Predicts n values in advance
    predict_stepwise(y, x=None, steps_ahead=1, y_start=None)
        Predicts a specified number of steps into the future for every time point in y-values array

    """

    def __init__(self,
                 n_nodes=1000, input_scaling=0.5, feedback_scaling=0.5, spectral_radius=0.8, leaking_rate=1.0,
                 connectivity=0.1, regularization=1e-8, feedback=True, random_seed=123):
        # Parameters
        self.n_nodes = int(np.round(n_nodes))
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.feedback = feedback
        self.seed = random_seed
        self.generate_reservoir()

    def generate_reservoir(self):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state
        random_state = np.random.RandomState(self.seed)

        # Set weights and sparsity randomly
        max_tries = 1000  # Will usually finish on the first iteration
        for i in range(max_tries):
            self.weights = random_state.uniform(-1., 1., size=(self.n_nodes, self.n_nodes))
            accept = random_state.uniform(size=(self.n_nodes, self.n_nodes)) < self.connectivity
            self.weights *= accept

            # Set spectral density
            max_eigenvalue = np.abs(np.linalg.eigvals(self.weights)).max()
            if max_eigenvalue > 0:
                break
            elif i == max_tries - 1:
                raise ValueError('Nilpotent reservoirs are not allowed. Increase connectivity and/or number of nodes.')

        # Set spectral radius of weight matrix
        self.weights *= self.spectral_radius / max_eigenvalue

        # Default state
        self.state = np.zeros((1, self.n_nodes), dtype=np.float32)

        # Set out to none to indicate untrained ESN
        self.out_weights = None

    def draw_reservoir(self):
        """Vizualizes reservoir.

        Requires 'networkx' package.

        """
        import networkx as nx
        graph = nx.DiGraph(self.weights)
        nx.draw(graph)

    def normalize(self, inputs=None, outputs=None, keep=False):
        """Normalizes array by column (along rows) and stores mean and standard devation.

        Set `store` to True if you want to retain means and stds for denormalization later.

        Parameters
        ----------
        inputs : array or None
            Input matrix that is to be normalized
        outputs : array or None
            Output column vector that is to be normalized
        keep : bool
            Stores the normalization transformation in the object to denormalize later

        Returns
        -------
        transformed : tuple or array
            Returns tuple of every normalized array. In case only one object is to be returned the tuple will be
            unpacked before returning

        """
        # Checks
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')

        # Storage for transformed variables
        transformed = []

        if not inputs is None:
            if keep:
                # Store for denormalization
                self._input_means = inputs.mean(axis=0)
                self._input_stds = inputs.std(ddof=1, axis=0)

            # Transform
            transformed.append((inputs - self._input_means) / self._input_stds)

        if not outputs is None:
            if keep:
                # Store for denormalization
                self._output_means = outputs.mean(axis=0)
                self._output_stds = outputs.std(ddof=1, axis=0)

            # Transform
            transformed.append((outputs - self._output_means) / self._output_stds)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

    def denormalize(self, inputs=None, outputs=None):
        """Denormalizes array by column (along rows) using stored mean and standard deviation.

        Parameters
        ----------
        inputs : array or None
            Any inputs that need to be transformed back to their original scales
        outputs : array or None
            Any output that need to be transformed back to their original scales

        Returns
        -------
        transformed : tuple or array
            Returns tuple of every denormalized array. In case only one object is to be returned the tuple will be
            unpacked before returning

        """
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')

        # Storage for transformed variables
        transformed = []

        if not inputs is None:
            transformed.append((inputs * self._input_stds) + self._input_means)
        if not outputs is None:
            transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

    def train(self, y, x=None, burn_in=100, input_weight=None):
        """Trains the Echo State Network.

        Trains the out weights on the random network. This is needed before being able to make predictions.
        Consider running a burn-in of a sizable length. This makes sure the state  matrix has converged to a
        'credible' value.

        Parameters
        ----------
        y : array
            Column vector of y values
        x : array or None
            Optional matrix of inputs (features by column)
        burn_in : int
            Number of inital time steps to be discarded for model inference

        Returns
        -------
        complete_data, y, burn_in : tuple
            Returns the complete dataset (state matrix concatenated with any feedback and/or inputs),
            the y values provided and the number of time steps used for burn_in. These data can be used
            for diagnostic purposes  (e.g. vizualization of activations).

        """
        # Checks
        if x is None and not self.feedback:
            raise ValueError("Error: provide x or enable feedback")

        # Initialize new random state
        random_state = np.random.RandomState(self.seed + 1)

        # Normalize inputs and outputs
        y = self.normalize(outputs=y, keep=True)
        if not x is None:
            x = self.normalize(inputs=x, keep=True)

        # Reset state
        current_state = self.state[-1]  # From default or pretrained state

        # Calculate correct shape based on feedback (feedback means one row less)
        start_index = 1 if self.feedback else 0  # Convenience index
        rows = y.shape[0] - start_index

        # Build state matrix
        self.state = np.zeros((rows, self.n_nodes), dtype=np.float32)

        # Build inputs
        inputs = np.ones((rows, 1), dtype=np.float32)  # Add bias for all t = 0, ..., T

        # Add data inputs if present
        if not x is None:
            inputs = np.hstack((inputs, x[start_index:]))  # Add data inputs

        # Set and scale input weights (for memory length and non-linearity)
        self.in_weights = self.input_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, inputs.shape[1]))

        # Add feedback if requested, optionally with feedback scaling
        if self.feedback:
            inputs = np.hstack((inputs, y[:-1]))  # Add teacher forced signal (equivalent to y(t-1) as input)
            feedback_weights = self.feedback_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, 1))
            self.in_weights = np.hstack((self.in_weights, feedback_weights))

        # Train iteratively
        for t in range(inputs.shape[0]):
            update = np.tanh(self.in_weights @ inputs[t].T + self.weights @ current_state)
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state  # Leaking separate
            self.state[t] = current_state

        # Concatenate inputs with node states
        complete_data = np.hstack((inputs, self.state))
        train_x = complete_data[burn_in:]  # Include everything after burn_in
        train_y = y[burn_in + 1:] if self.feedback else y[burn_in:]

        # Ridge regression
        ridge_x = train_x.T @ train_x + self.regularization * np.eye(train_x.shape[1])
        ridge_y = train_x.T @ train_y

        # Full inverse solution
        # self.out_weights = np.linalg.inv(ridge_x) @ ridge_y

        # Solver solution (fast)
        self.out_weights = np.linalg.solve(ridge_x, ridge_y)

        # Store last y value as starting value for predictions
        self.y_last = y[-1]

        # Return all data for computation or visualization purposes (Note: these are normalized)
        return complete_data, (y[1:] if self.feedback else y), burn_in

    def test(self, y, x=None, y_start=None, steps_ahead=None, scoring_method='mse', alpha=1.):
        """Tests and scores against known output.

        Parameters
        ----------
        y : array
            Column vector of known outputs
        x : array or None
            Any inputs if required
        y_start : float or None
            Starting value from which to start testing. If None, last stored value from trainging will be used
        steps_ahead : int or None
            Computes average error on n steps ahead prediction. If `None` all steps in y will be used.
        scoring_method : {'mse', 'rmse', 'nrmse', 'tanh'}
            Evaluation metric used to calculate error
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}

        Returns
        -------
        error : float
            Error between prediction and knwon outputs

        """
        # Run prediction
        final_t = y.shape[0]
        if steps_ahead is None:
            y_predicted = self.predict(final_t, x, y_start=y_start)
        else:
            y_predicted = self.predict_stepwise(y, x, steps_ahead=steps_ahead, y_start=y_start)[:final_t]

        # Return error
        return self.error(y_predicted, y, scoring_method, alpha=alpha)

    def predict(self, n_steps, x=None, y_start=None):
        """Predicts n values in advance.

        Prediction starts from the last state generated in training.

        Parameters
        ----------
        n_steps : int
            The number of steps to predict into the future (internally done in one step increments)
        x : numpy array or None
            If prediciton requires inputs, provide them here
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value from training will be used

        Returns
        -------
        y_predicted : numpy array
            Array of n_step predictions

        """
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')

        # Normalize the inputs (like was done in train)
        if not x is None:
            x = self.normalize(inputs=x)

        # Initialize input
        inputs = np.ones((n_steps, 1), dtype=np.float32)  # Add bias term
        
        print("inputs_shape: " + str(inputs.shape))
        # Choose correct input
        if x is None and not self.feedback:
            raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
        elif not x is None:
            inputs = np.hstack((inputs, x))  # Add data inputs

        # Set parameters
        y_predicted = np.zeros(n_steps, dtype=np.float32)

        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]

        # Initialize state from last availble in train
        current_state = self.state[-1]

        # Predict iteratively
        for t in range(n_steps):
            # Get correct input based on feedback setting
            current_input = inputs[t] if not self.feedback else np.hstack((inputs[t], previous_y))

            # Update
            update = np.tanh(self.in_weights @ current_input.T + self.weights @ current_state)
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state

            # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
            complete_row = np.hstack((current_input, current_state))
            y_predicted[t] = complete_row @ self.out_weights
            previous_y = y_predicted[t]

        # Denormalize predictions
        y_predicted = self.denormalize(outputs=y_predicted)

        # Return predictions
        return y_predicted.reshape(-1, 1)

    def predict_stepwise(self, y, x=None, steps_ahead=1, y_start=None):
        """Predicts a specified number of steps into the future for every time point in y-values array.

        E.g. if `steps_ahead` is 1 this produces a 1-step ahead prediction at every point in time.

        Parameters
        ----------
        y : numpy array
            Array with y-values. At every time point a prediction is made (excluding the current y)
        x : numpy array or None
            If prediciton requires inputs, provide them here
        steps_ahead : int (default 1)
            The number of steps to predict into the future at every time point
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value from training will be used

        Returns
        -------
        y_predicted : numpy array
            Array of predictions at every time step of shape (times, steps_ahead)

        """
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')

        # Normalize the arguments (like was done in train)
        y = self.normalize(outputs=y)
        if not x is None:
            x = self.normalize(inputs=x)

        # Timesteps in y
        t_steps = y.shape[0]

        # Check input
        if not x is None and not x.shape[0] == t_steps:
            raise ValueError('x has the wrong size for prediction: x.shape[0] = {}, while y.shape[0] = {}'.format(
                x.shape[0], t_steps))

        # Choose correct input
        if x is None and not self.feedback:
            raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
        elif not x is None:
            # Initialize input
            inputs = np.ones((t_steps, 1), dtype=np.float32)  # Add bias term
            inputs = np.hstack((inputs, x))  # Add x inputs
        else:
            # x is None
            inputs = np.ones((t_steps + steps_ahead, 1), dtype=np.float32)  # Add bias term

        # Run until we have no further inputs
        time_length = t_steps if x is None else t_steps - steps_ahead + 1

        # Set parameters
        y_predicted = np.zeros((time_length, steps_ahead), dtype=np.float32)

        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]

        # Initialize state from last availble in train
        current_state = self.state[-1]

        # Predict iteratively
        for t in range(time_length):

            # State_buffer for steps ahead prediction
            prediction_state = np.copy(current_state)

            # Y buffer for step ahead prediction
            prediction_y = np.copy(previous_y)

            # Predict stepwise at from current time step
            for n in range(steps_ahead):

                # Get correct input based on feedback setting
                prediction_input = inputs[t + n] if not self.feedback else np.hstack((inputs[t + n], prediction_y))

                # Update
                prediction_update = np.tanh(self.in_weights @ prediction_input.T + self.weights @ prediction_state)
                prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state

                # Store for next iteration of t (evolves true state)
                if n == 0:
                    current_state = np.copy(prediction_state)

                # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
                prediction_row = np.hstack((prediction_input, prediction_state))
                y_predicted[t, n] = prediction_row @ self.out_weights
                prediction_y = y_predicted[t, n]

            # Evolve true state
            previous_y = y[t]

        # Denormalize predictions
        y_predicted = self.denormalize(outputs=y_predicted)

        # Return predictions
        return y_predicted

    def error(self, predicted, target, method='mse', alpha=1.):
        """Evaluates the error between predictions and target values.

        Parameters
        ----------
        predicted : array
            Predicted value
        target : array
            Target values
        method : {'mse', 'tanh', 'rmse', 'nmse', 'nrmse', 'tanh-nmse', 'log-tanh', 'log'}
            Evaluation metric. 'tanh' takes the hyperbolic tangent of mse to bound its domain to [0, 1] to ensure
            continuity for unstable models. 'log' takes the logged mse, and 'log-tanh' takes the log of the squeezed
            normalized mse. The log ensures that any variance in the GP stays within bounds as errors go toward 0.
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}.
            This squeezes errors onto the interval [0, alpha].
            Default is 1. Suggestions for squeezing errors > n * stddev of the original series
            (for tanh-nrmse, this is the point after which difference with y = x is larger than 50%,
             and squeezing kicks in):
             n  |  alpha
            ------------
             1      1.6
             2      2.8
             3      4.0
             4      5.2
             5      6.4
             6      7.6

        Returns
        -------
        error : float
            The error as evaluated with the metric chosen above

        """
        # Return error based on choices
        if predicted.shape[1] == 1:
            errors = predicted.ravel() - target.ravel()
        else:
            # Multiple prediction columns
            errors = np.zeros(predicted.shape, dtype=np.float32)
            for i in steps_ahead:
                predictions = predicted[:, i]
                errors[:, i] = predictions.ravel()[:-i] - target.ravel()[i:]

        # Adjust for NaN and np.inf in predictions (unstable solution)
        if not np.all(np.isfinite(predicted)):
            # print("Warning: some predicted values are not finite")
            errors = np.inf

        # Compute mean error
        if method == 'mse':
            error = np.mean(np.square(errors))
        elif method == 'tanh':
            error = alpha * np.tanh(np.mean(np.square(errors)) / alpha)  # To 'squeeze' errors onto the interval (0, 1)
        elif method == 'rmse':
            error = np.sqrt(np.mean(np.square(errors)))
        elif method == 'nmse':
            error = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
        elif method == 'nrmse':
            error = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
        elif method == 'tanh-nrmse':
            nrmse = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
            error = alpha * np.tanh(nrmse / alpha)
        elif method == 'log':
            mse = np.mean(np.square(errors))
            error = np.log(mse)
        elif method == 'log-tanh':
            nrmse = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
            error = np.log(alpha * np.tanh((1. / alpha) * nrmse))
        else:
            raise ValueError('Scoring method not recognized')
        return error
