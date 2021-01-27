import numpy as np
import scipy.stats
import scipy.linalg
import copy
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

__all__ = ['EchoStateNetwork']

colorz = {
  "header" : '\033[95m',
  "blue" : '\033[94m',
  'cyan' : '\033[96m',
  'green' : '\033[92m',
  'warning' : '\033[93m',
  'fail' : '\033[91m',
  'endc' : '\033[0m',
   'bold' :'\033[1m',
   "underline" : '\033[4m'
}
  
def printc(string_, color_) :
  print(colorz[color_] + string_ + colorz["endc"] )

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import time


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
                 connectivity = np.exp(-.23),
                 regularization=1e-8, feedback=False, random_seed=123,
                 activation_function = "tanh",
                 exponential=False, llambda = None, 
                 llambda2 = None, model_type = None, noise = 0.0, 
                 obs_idx = None, plot = False, resp_idx = None, 
                 cyclic_res_w = None, 
                 cyclic_input_w = None, 
                 cyclic_bias = None,
                 already_normalized = False,
                 input_weight_type = "uniform",
                 Distance_matrix = None,
                 reservoir = None,
                 input_weights_from_cv = None,
                 reservoir_output = None,
                 approximate_reservoir = True,
                 bias_scaling = 0.5

                 ):
        self.bias_scaling = bias_scaling
        # Parameters
        self.input_weights_from_cv = input_weights_from_cv
        self.reservoir = reservoir
        self.Distance_matrix = Distance_matrix
        self.input_weight_type = input_weight_type
        self.n_nodes = int(np.round(n_nodes))
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.feedback = feedback
        if not random_seed:
            self.seed = np.random.randint(100000)
        else:
            self.seed = random_seed
        self.approximate_reservoir = approximate_reservoir

        self.obs_idx = obs_idx
        self.resp_idx = resp_idx
        self.llambda = llambda
        self.plot = plot
        self.noise = noise
        self.already_normalized = already_normalized            

        self.model_type = model_type
        assert self.model_type, "you must choose a model"
        
        if llambda2:
            self.llambda = [self.llambda, llambda2]
            self.dual_lambda = True
        else:
            self.dual_lambda = False

        assert self.model_type in ["random", "delay_line", "cyclic"], self.model_type + str(" not implimented")
        assert self.input_weight_type in ["exponential", "uniform"], self.input_weight_type + str(" not implimented")


        #reservoir generation:
        if self.model_type in ["random"]:
            #random RC reservoir
            self.generate_reservoir()

        elif self.model_type in ["delay_line", "cyclic"]:
            self.cyclic_res_w  = cyclic_res_w
            self.cyclic_input_w = cyclic_input_w
            self.cyclic_bias = cyclic_bias 

            #cyclic / delay line RC reservoir
            self.generate_delay_line()

        #activation function: (sin_sq if delay line)
        def sin_sq(arr):
            temp = np.sin(arr)
            return(temp**2)

        assert activation_function in ["tanh", "sin_sq"]

        if activation_function == "tanh":
            self.activation_function = np.tanh
        elif activation_function == "sin_sq":
            self.activation_function = sin_sq
    

    def exp_w(self,  random_state, n_inputs, verbose = False):
        """
        Args:
            llambda: is llambda in an exponential function.
            distance: is a distance matrix. 
        
        This function calculates weights via attention ie the
        distance matrix which measures the distance from the 
        observer sequences to the target sequences.
        """
        random_state = np.random.RandomState(self.seed)
        random_state2 = np.random.RandomState(self.seed +1)
        
        def get_exp_w_set(llambda_, distance_np_):
            exp_np_ = np.exp( - llambda_ * distance_np_) #*llambda
            exp_np_ = exp_np_.sum(axis = 0).reshape(-1, )
            
            #normalize the max weight to 1.
            exp_np_ = exp_np_ / np.max(exp_np_)

            return(exp_np_)


        if not self.dual_lambda:
            
            exp_np = get_exp_w_set( self.llambda, self.distance_np)
            #white_Noise = random_state.normal(loc = 0, scale = self.noise, size = (self.n_nodes, exp_np.shape[0]))
            #exp_np = exp_np + white_Noise
            sign = random_state.choice([-1, 1], self.exp_weights.shape)

            if self.llambda < 10 ** (-4):
                exp_np = random_state.uniform(-1, 1, size=(exp_np.shape)) * sign
            
        else:
            exp_np = np.ones(shape=(self.n_nodes, n_inputs))
            nn = self.distance_np[0].shape[1]
            
            
            if self.reservoir:
                sign = self.reservoir.sign_dual
            else:
                sign1 = random_state.choice([-1, 1], size= (self.n_nodes, self.distance_np[0].shape[1]))
                sign2 = random_state2.choice([-1, 1], size= (self.n_nodes, self.distance_np[1].shape[1]))
                sign = (sign1, sign2)

            ### There are speed gains to be made at this point. You can pre-load the random_uniform arrays.
            print("exp_np", exp_np.shape)

            if self.llambda[0] < 10 ** (-4):
                exp_np[:, :nn] = random_state.uniform(-1, 1, size=(self.n_nodes, self.distance_np[0].shape[1]))
            else:
                exp_np1 = get_exp_w_set( self.llambda[0], self.distance_np[0])
                print("exp_np1", exp_np1.shape)
                exp_np1 = exp_np1 * sign[0]
                #printc("exp_np1: " + str(exp_np1.shape), 'fail')
                exp_np[:, :nn] =  exp_np1

            if self.llambda[1] < 10 ** (-4):
                exp_np[:, nn:] = random_state.uniform(-1, 1, size=(self.n_nodes, self.distance_np[1].shape[1])) 
            else:           
                exp_np2 = get_exp_w_set( self.llambda[1], self.distance_np[1])
                exp_np2 = exp_np2 * sign[1]

                exp_np[:, nn:] = exp_np2
                #printc("exp_np2: " + str(exp_np2.shape), 'fail')
            
            #this used to be where we added noise:
            #white_Noise = random_state.normal(loc = 0, scale = self.noise, size = (self.n_nodes, exp_np.shape[0]))
        self.exp_weights = exp_np

       
    def build_distance_matrix(self, verbose = False):
        """	
        args:
            resp is the response index (a list of integers associated with the target train/test time series 
                (for example individual frequencies)
            obs is the same for the observation time-series.
        Description:
        	DistsToTarg stands for distance numpy array
        """
        def calculate_distance_matrix(obs_idx):
            #assert 2 ==1, "DON'T CALCULATE THIS INTERNALLY IF YOU CAN AVOID IT"
            obs_idxx_arr = np.array(obs_idx)
            for i, resp_seq in enumerate(self.resp_idx):
                DistsToTarg = abs(resp_seq - obs_idxx_arr).reshape(1, -1)
                if i == 0:
                    distance_np_ = DistsToTarg
                else:
                    distance_np_ = np.concatenate([distance_np_, DistsToTarg], axis = 0)
            distance_np_ = distance_np_
            if verbose == True:
                print("distance_matrix completed " + str(self.distance_np_.shape))
                display(self.distance_np_)
            return(distance_np_)

        if not self.dual_lambda:

            self.distance_np = calculate_distance_matrix(self.obs_idx) 
            if self.plot:
                plt.imshow(self.distance_np)
                plt.show()
        else:
            def split_lst(lst, scnd_lst):
                
                lst = np.array(lst)
                breaka = np.mean(scnd_lst)
                scnd_arr = np.array(scnd_lst)
                lst1, lst2 = lst[lst < scnd_arr.min()], lst[lst > scnd_arr.max()]

                return list(lst1), list(lst2)

            obs_lsts = split_lst(self.obs_idx, self.resp_idx) #good!
            self.distance_np = [calculate_distance_matrix(obs_lst) for obs_lst in obs_lsts]
            if self.plot:
                for i in self.distance_np:
                    plt.imshow(i)
                    plt.show()

    def get_exp_weights(self, n_inputs):
        """
        #TODO: description
        change the automatic var assignments
        """
        if type(self.Distance_matrix) == type(None):
            print("building distance matrix, you should externalize this if CV")
            self.build_distance_matrix()
        else:
            self.distance_np = self.Distance_matrix
        random_state = np.random.RandomState(self.seed)
        self.exp_w(random_state, n_inputs)

        

        if self.plot == True:
            print("PLOTTING")
            for i in range(self.exp_weights.shape[0]):
                if not i:
                    exp_w_dict = {}
                    exp_w_dict["obs_idx"] = []
                    exp_w_dict["weight"] = []
                #lst.append({"obs_idx": , "weight": })
                exp_w_dict["weight"]  += list(self.exp_weights[i, :].reshape(-1,))
                exp_w_dict["obs_idx"] += self.obs_idx

            pd_ = pd.DataFrame(exp_w_dict)
            #print(pd_)
            fig, ax = plt.subplots(1, 1, figsize = (6, 4))
            sns.scatterplot(x = "obs_idx", y = "weight", data = pd_, ax = ax, linewidth=0, alpha = 0.003)
            ax.set_title("Exponential Attention Weights")
            plt.show()
    
    def generate_reservoir(self, obs_idx = None, targ_idx = None, load_failed = None):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state
        start = time.time()
        random_state = np.random.RandomState(self.seed)
        #assert type(load_failed) == type(None)
        # Set weights and sparsity randomly
        max_tries = 1000  # Will usually finish on the first iteration
        n = self.n_nodes

        #if the size of the reservoir has changed, reload it. RENAME THE VARIABLE
        if self.reservoir:
            if self.reservoir.n_nodes_ != self.n_nodes:
                load_failed = 1
        
        book_index = 0
        for i in range(max_tries):
            if i > 0:
                printc(str(i), 'fail')
            #only initialize the reservoir and connectivity matrix if we have to for speed in esn_cv.
            if not self.reservoir or not self.approximate_reservoir or load_failed == 1:

                if i <= 30:
                    self.weights = random_state.uniform( -1., 1., size = (n, n))
                    self.accept = np.random.uniform(size = (n, n))  < self.connectivity
                    #TODO SANITY CHECK THAT THIS CHANGES
                	#elif i < 30:
                	#    #Try varying acceptance adjacency matrices.
                	#    self.accept = np.random.uniform(size = (n, n))  < self.connectivity
                else:
                	#otherwise create a completely random set.
                	print("Randomizing CONNECTIONS FAILED, Randomizing reservoir")
                	self.weights = np.random.uniform( -1., 1., size = (n, n))
                	self.accept = np.random.uniform(size = (n, n))  < self.connectivity
                
                self.weights *= self.accept
                self.weights = csc_matrix(self.weights)
            else:
                #print("LOADING MATRIX", load_failed)
                if self.approximate_reservoir:
                    try:   
                        self.weights = self.reservoir.get_approx_preRes(self.connectivity, i) #np.random.choice([0,1]))
                        #printc("reservoir successfully loaded (" + str(self.weights.shape) , 'green') 
                    except:
                        printc("approx reservoir " + str(i) + " failed to load... regenerating", 'fail')
                        #print("loading_approx_res")
                        #skip to the next iteration of the loop
                        if i > self.reservoir.number_of_preloaded_sparse_sets:
                        	load_failed = 1
                        	printc("All preloaded reservoirs Nilpotent, generating random reservoirs." + ", connectivity =" + str(round(self.connectivity,8)) + '...regenerating', 'fail')
                
                        continue
                        #self.weights = self.reservoir.get_approx_preRes(self.connectivity, i)
                    	
   
                else:
                    assert 1 == 0, "TODO, case not yet handled."

 
            try:

                eigs_list = eigs(self.weights, k = 1, which  = 'LM', return_eigenvectors = False)

                max_eigenvalue = np.abs(eigs_list)[0]
            except:
                max_eigenvalue = np.abs(np.linalg.eigvals(self.weights.toarray())).max()

            if max_eigenvalue > 0:
                
                break
            else:
                printc("Loaded Reservoir is Nilpotent (max_eigenvalue =" + str(max_eigenvalue) + "), connectivity =" + str(round(self.connectivity,8))  + '...regenerating', 'fail')
                
                #if we have run out of pre-loaded reservoirs to draw from :

                if i == max_tries - 1:
                
                    raise ValueError('Nilpotent reservoirs are not allowed. Increase connectivity and/or number of nodes.')

        # Set spectral radius of weight matrix
        self.weights = csc_matrix(self.weights*  (self.spectral_radius / max_eigenvalue ))#csc_matrix(self.weights)

        if load_failed == 1 or not self.reservoir:
            # Default state #CAN BE DEEP-COPIED
            self.state = np.zeros((1, self.n_nodes), dtype=np.float32)
        else:
            #print("loading state", self.reservoir.state[0:5])
            self.state = self.reservoir.state

        # Set out to none to indicate untrained ESN
        self.out_weights = None
        #printc('TIME TO gen_res: ' + str(time.time() - start), 'green')

    def generate_delay_line(self):
        """Generates the simple delay line reservoir from ?parameters? set at initialization."""

        # Initialize new random state
        #random_state = np.random.RandomState(self.seed)
        n = self.n_nodes
        self.weights = np.zeros(shape = (n, n))

        for i in range(n - 1):
            #weights[i + 1, i] = cyclic_weight
            self.weights[ i + 1, i] = self.cyclic_res_w

        # Default state
        self.state = np.zeros((1, self.n_nodes), dtype=np.float32)

        # Set out to none to indicate untrained ESN
        self.out_weights = None

        # This is the only difference between the cyclic reservoir and the delay line.
        if self.model_type == "cyclic":
            self.weights[0, -1] = self.cyclic_res_w


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
                # Store for denormalizationf
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


    def train(self, y, x=None, burn_in=0, input_weight=None, verbose = False):
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

        start = time.time()
        #verbose = True
        if x is None and not self.feedback:
            self.already_normalized = True
            self.n_inputs = y.shape[1]
            x = np.ones((y.shape[0], y.shape[1] -1))
            #self.x = x.copy()
            #raise ValueError("Error: provide x or enable feedback")

        # Initialize new random state
        random_state = np.random.RandomState(self.seed + 1)

        if not self.already_normalized:
            # Normalize inputs and outputs
            y = self.normalize(outputs=y, keep=True)
            if not x is None:
                x = self.normalize(inputs=x, keep=True)
        self.y = y.copy()

        # Reset state
        current_state = self.state[-1]  # From default or pretrained state

        # Calculate correct shape based on feedback (feedback means one row less)
        start_index = 1 if self.feedback else 0  # Convenience index
        #print('y.shape',y.shape)
        rows = y.shape[0] - start_index

        # Build state matrix
        self.state = np.zeros((rows, self.n_nodes), dtype=np.float32)

        # Build inputs
        inputs = np.ones((rows, 1), dtype=np.float32)  # Add bias for all t = 0, ..., T

        # Add data inputs if present
        if not x is None:
            inputs = np.hstack((inputs, x[start_index:]))  # Add data inputs

        if not self.reservoir or 'in_weights' not in dir(self.reservoir): 
            
            if load_failed == 1 or not self.reservoir:
            # Default state #CAN BE DEEP-COPIED
            self.state = np.zeros((1, self.n_nodes), dtype=np.float32)
        else:
            #print("loading state", self.reservoir.state[0:5])
            self.state = self.reservoir.state
            if self.input_weight_type in ["exponential", "uniform"]:
                n_inputs = (y.shape[1] - 1) if x is None else x.shape[1]
                self.n_inputs = n_inputs
                print('in weight type', self.input_weight_type)
                if self.input_weight_type == "exponential":
                    #print("EXP")
                    self.get_exp_weights(x.shape[1])
                    
                    if self.reservoir:
                        uniform_bias = self.reservoir.uniform_bias
                        #print("BUILDING EXPONENTIAL IN WEIGHTS" )
                    else:
                        uniform_bias = random_state.uniform(-1, 1, size = (self.n_nodes, 1))
                    self.in_weights =  self.exp_weights
                else:
                    print("BUILDING UNIFORM IN WEIGHTS")
                    
                    self.in_weights = random_state.uniform(-1, 1, size=(self.n_nodes, n_inputs))
                    uniform_bias = random_state.uniform(-1, 1, size = (self.n_nodes, 1))

                
                 #* self.bias_scaling
                if self.noise:
                    white_noise = random_state.normal(loc = 0, scale = self.noise, size = (self.n_nodes, n_inputs)) #self.in_weights.shape[1] - 1
                    #print(self.in_weights.shape, "in_weights")
                    #print(white_noise.shape, "white noise")
                    self.in_weights += white_noise
                self.in_weights = np.hstack((uniform_bias, self.in_weights)) * self.input_scaling

                #self.reservoir.in_weights = self.in_weights

            elif self.input_weight_type == "cyclic":

                # Set and scale input weights (for memory length and non-linearity)
                self.in_weights = np.full(shape=(self.n_nodes, inputs.shape[1] - 1), fill_value=self.cyclic_input_w, dtype=np.float32)
                self.in_weights *= np.sign(random_state.uniform(low=-1.0, high=1.0, size=self.in_weights.shape))
                self.in_weights *= self.input_scaling 
                
                #add input bias
                cyclic_bias = np.full(shape=(self.n_nodes, 1), fill_value=self.cyclic_bias, dtype=np.float32)
                cyclic_bias *= np.sign(random_state.uniform(low=-1.0, high=1.0, size=self.cyclic_bias.shape))
                
                self.in_weights = np.hstack((cyclic_bias, self.in_weights)) 
                self.input_weights_from_cv = self.in_weights
        else:
            #print("loading in_weights from cv.")
            self.in_weights = self.reservoir.in_weights + self.noise * self.reservoir.noise_z


        if self.model_type == "delay_line":
            """
            SANDBOX RESULT:

            beta = 3
            alpha = 0.5
            n_nodes = 5
            n_inputs = 4
            input_bias = np.full((1,), fill_value = alpha)
            input_weights = np.full((n_inputs,), fill_value = beta)

            input_weights = np.hstack((input_bias, input_weights))
            input_weights_zeroes = np.zeros((n_nodes-1, n_inputs + 1))

            input_weights = np.vstack((input_weights, input_weights_zeroes))
            display(input_weights)
            orig_inputs = np.random.uniform(-1, 1, size = 4)
            inputs = np.hstack((1, orig_inputs))
            print(inputs)
            print(alpha + np.sum(orig_inputs)*beta)
            input_weights @ inputs.T
            """
            #Build input weights matrix:
            bias_phi = True


            #Bias term like phi
            if bias_phi:
                input_weight = np.full( shape = (inputs.shape[1] - 1, ), fill_value = self.cyclic_input_w, dtype=np.float32)
                input_weights_zeroes = np.zeros((self.n_nodes - 1, inputs.shape[1] -1 ))
                in_weights = np.vstack((input_weight, input_weights_zeroes))
                input_bias = np.full( shape = (self.n_nodes, ), fill_value = self.cyclic_bias, dtype=np.float32).reshape(-1,1)
                self.in_weights = np.hstack((input_bias, in_weights))
            #original bias
            else:
                input_weight = np.full( shape = (inputs.shape[1] - 1, ), fill_value = self.cyclic_input_w, dtype=np.float32)
                input_bias = np.full( shape = (1, ), fill_value = self.cyclic_bias, dtype=np.float32)
                input_weights = np.hstack((input_bias, input_weight))
                # add zeros
                input_weights_zeroes = np.zeros((self.n_nodes - 1, inputs.shape[1]))
                self.in_weights = np.vstack((input_weights, input_weights_zeroes))
            self.in_weights *= self.input_scaling

        # Add feedback if requested, optionally with feedback scaling
        if self.feedback:
            
            #1by3153 X 3153 by 1000 --> 1by1000
            #1by1000 X 1000 by 3153 --> 
            #2by3153 X (1000by2) --> 2by3153

            last_y = y[:-1,:]#.reshape(-1,rows)
            
            #print("last_y", y.shape)
            #print("y[-1]", y[-1,:].shape)

            inputs = np.hstack((inputs, last_y )) # Add teacher forced signal (equivalent to y(t-1) as input)
            #print("inputs", inputs.shape)
            feedback_weights = self.feedback_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, 1))
            #feedback_weights = self.feedback_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, inputs.shape[1] - 1))
            self.in_weights = np.hstack((self.in_weights, feedback_weights)).reshape(self.n_nodes, -1)
            #print("in_weights", self.in_weights.shape)

        # Train iteratively
        #vverbose = 0
        for t in range(inputs.shape[0]):

            #for debugging:
            #if verbose:
            #if vverbose == 2:
	        #    print("in_weights: "  + str(self.in_weights.shape))
	        #    print("inputs[t].T: " + str(inputs[t].T.shape))
            #
            # #res = self.in_weights @ inputs[t].T
            # #if vverbose == 2:
	        # #    print("res: " + str(res.shape))
	        # #    print("current_state: " + str(current_state.shape))
	        # #    print("weights: " + str(self.weights.shape))
            # #hi = self.weights.dot(current_state)
            # #if vverbose == 2:
            #	print("hi",hi.shape)
            
            #for debugging
            update = self.activation_function(self.in_weights @ inputs[t].T + self.weights.dot(current_state))
            
            #if verbose:    
            
            #print("update: " + str(update.shape))
            
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state  # Leaking separate
            self.state[t] = current_state

        #printc('TIME TO TRAIN: ' + str(time.time() - start), 'green')

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
        self.y_last = y[-1, :]

        # Return all data for computation or visualization purposes (Note: these are normalized)
        return complete_data, (y[1:,:] if self.feedback else y), burn_in

    def test(self, y, x=None, y_start=None, steps_ahead=None, scoring_method='nmse', alpha=1.):
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
        
        if steps_ahead is None:
            y_predicted = self.predict(n_steps = y.shape[0], x=x, y_start=y_start)
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
            y_predicted = self.predict_stepwise(y, x, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]

        # Return error
        return self.error(y_predicted, y, scoring_method, alpha=alpha)

    def print_version(self):
        print("previous y")

    def predict(self, n_steps, pure_prediction = True, x=None, y_start=None):
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

        inputs = np.ones((n_steps, 1), dtype=np.float32)  # Add bias term


        if not self.already_normalized:
            # Normalize the inputs (like was done in train)
            if not x is None:
                x = self.normalize(inputs=x)


        #printc("n_steps: " + str(n_steps), 'fail')

        if x is None and not self.feedback:
            #self.already_normalized = True
            inputs = np.ones((n_steps, self.in_weights.shape[1]))
            #printc("solving pred without feedback: x.shape: " + str(x.shape), 'fail')
        #    #raise ValueError("Error: provide x or enable feedback")
        #    inputs = np.ones((self.y.shape[0], 1), dtype=np.float32)  # Add bias term
        #else:
        #	
        
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')

        

        # Initialize input
        
        

        # Choose correct input
        #if x is None and not self.feedback:
        #    raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
        #print("inputs", inputs.shape)
        #print("x", x.shape)
        if not x is None and self.feedback: 
            inputs = np.hstack((inputs, x)) 
        elif x is not None and not self.feedback:
        	inputs = np.hstack((inputs, x)) 
        	# Add data inputs

        # Set parameters
        if self.out_weights.shape[1] > 1:
          y_predicted = np.zeros([n_steps, self.out_weights.shape[1]], dtype=np.float32)
        else:
          y_predicted = np.zeros(n_steps, dtype=np.float32)

        previous_y = self.y_last

        if not self.already_normalized:
          # Get last states
          
          if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]

        # Initialize state from last availble in train
        current_state = self.state[-1]



        # Predict iteratively
        for t in range(n_steps):
            #print("current_input.T: " + str(inputs[t].T.shape))
            # Get correct input based on feedback setting
            current_input = inputs[t].T if not self.feedback else np.hstack((inputs[t], previous_y))

            # Update

            #print("in_weights: "  + str(self.in_weights.shape))
            #print("current_input: " + str(current_input.T.shape))
            #res = self.in_weights @ inputs[t].T
            #print("res: " + str(res.shape))

            
            #print("current_state: " + str(current_state.shape))
            #print("weights: " + str(self.weights.shape))
            #hi = self.weights.dot(current_state)
            #print("hi",hi.shape)


            update = self.activation_function(self.in_weights @ current_input + self.weights @ current_state)
            #print("update: " + str(update.shape))
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state

            # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
            complete_row = np.hstack((current_input, current_state))

            if self.out_weights.shape[1] > 1:
              y_predicted[t,:] = complete_row @ self.out_weights
              previous_y = y_predicted[t,:]
            else:
              y_predicted[t] = complete_row @ self.out_weights
              #print("t: ", t, "y_predicted", y_predicted[t])
              previous_y = y_predicted[t]
              #print("t: ", t, "y_predicted", y_predicted[t].shape)

            
        if not self.already_normalized:
            # Denormalize predictions
            y_predicted = self.denormalize(outputs=y_predicted)

        return y_predicted.reshape(-1, self.out_weights.shape[1])

    def predict_stepwise(self, y, x=None, steps_ahead=1, y_start=None):
        """Predicts a specified number of steps into the future for every time point in y-values array.

        E.g. if `steps_ahead` is 1 this produces a 1-step ahead prediction at every point in time.

        Parameters
        ----------
        y : numpy array
            Array with y-values. At every time point a prediction is made (excluding the current y)
        x : numpy array or None
            If prediction requires inputs, provide them here
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

        if not self.already_normalized:
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
        if not self.already_normalized:
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

                #res = self.in_weights @ inputs[t].T

                print("prediction_input", prediction_row.shape)
                print("predicted_state", prediction_state)

                # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
                prediction_row = np.hstack((prediction_input, prediction_state))
                pred = prediction_row @ self.out_weights
                print("prediction_row", prediction_row.shape)
                print("y_predicted", y_predicted.shape)
                print("pred", pred.shape)
                y_predicted[:, n] = pred.reshape(-1,1)
                prediction_y = y_predicted[t, n]

            # Evolve true state
            previous_y = y[t]

        if not self.already_normalized:
            # Denormalize predictions
            y_predicted = self.denormalize(outputs=y_predicted)

        # Return predictions
        return y_predicted

    def error(self, predicted, target, method='nmse', alpha=1.):
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
        errors = predicted - target
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
