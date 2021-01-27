from .esn import *
from .scr import *
from .detail.robustgpmodel import *
from .detail.esn_bo import *
import numpy as np
import GPy
import GPyOpt
import copy
import json
import pyDOE
from collections import OrderedDict
import multiprocessing


__all__ = ['EchoStateNetworkCV']

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

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

def printc(string_, color_) :
  print(colorz[color_] + string_ + colorz["endc"] )

#from scipy.sparse import csr_matrix
#>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5

class SparseBooklet:
    def __init__(self, book, keys):
        self.sparse_book = book
        self.sparse_keys_ = keys

    def get_approx_preRes(self, connectivity_threshold):
        """
        You can use the matrix returned instead of...
        """
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  self.sparse_book[key_].copy()
        return val

class GlobalSparseLibrary:

    def __init__(self, lb = -5, ub = 0, n_nodes = 1000, precision = None, flip_the_script = False):
        self.lb = lb
        self.ub = ub
        self.n_nodes_ = n_nodes
        self.library = {}
        self.book_indices = []
        self.precision = precision
        self.flip_the_script = flip_the_script



    def addBook(self, random_seed):
        """
        Add a sparse reservoir set.
        """
        book = {}
        random_state = torch.Generator().manual_seed(random_seed)
        n = self.n_nodes_

        accept = torch.FloatTensor(n, n).uniform_(-1, 1, generator = random_state) 
        reservoir_pre_weights = torch.FloatTensor(n, n).uniform_(-1, 1, generator = random_state)

        "for now we're going to avoid sparse matrices"
        for connectivity in np.logspace(self.ub, self.lb, self.precision):
            #book[connectivity] = csc_matrix((accept < connectivity ) * reservoir_pre_weights)
            
            book[connectivity] = (accept < connectivity ) * reservoir_pre_weights
        sparse_keys_ = np.array(sorted(book))

        self.library[random_seed] = SparseBooklet(book = book, keys = sparse_keys_)
        self.book_indices.append(random_seed)

    def getIndices(self):
        return self.book_indices

    def get_approx_preRes(self, connectivity_threshold, index = 0):
        """
        You can use the matrix returned instead of...
        """
        if self.flip_the_script:
            index = np.random.randint(len(self.book_indices))
        book = self.library[self.book_indices[index]]
        if index != 0:
            printc("retrieving book from library" + str(self.book_indices[index]), 'green')
        return book.get_approx_preRes(connectivity_threshold)


class ReservoirBuildingBlocks:
    """ An object that allows us to save reservoir components (independent of hyper-parameters) for faster optimization.
    Parameters:
        model_type: either random, cyclic or delay line
        input_weight_type: exponential or uniform
        random_seed: the random seed to set the reservoir
        n_nodes: the nodes of the network
        n_inputs: the number of observers in the case of a block experiment, the size of the output in the case of a pure prediction where teacher forcing is used.

    """
    def __init__(self, model_type, input_weight_type, random_seed, n_nodes, n_inputs = None, Distance_matrix = None):
        #print("INITIALING RESERVOIR")

        #initialize attributes
        self.input_weight_type_ = input_weight_type
        self.model_type_ = model_type
        self.n_inputs_ = n_inputs
        self.n_nodes_ = n_nodes
        self.seed_ = random_seed
        self.state = np.zeros((1, self.n_nodes_), dtype=np.float32)
        
        if model_type == "random":
            self.gen_ran_res_params()
            self.gen_sparse_accept_dict()

    def gen_ran_res_params(self):
        random_state = torch.Generator().manual_seed(self.seed_)
        n = self.n_nodes_
        self.accept = torch.FloatTensor(n, n).uniform_(-1, 1, generator = random_state)
        self.reservoir_pre_weights = torch.FloatTensor(n, n).uniform_(-1, 1, generator = random_state)

    def gen_sparse_accept_dict(self, reservoir_seeds = [123, 999], precision = 1000):
        """
        Later change this so that you put in the real search bounds.
        This will approximate the search for the sparcity hyper-parameter, which will dramatically speed up training of the network.
        Later we can add in a final stage where the network does approximate sparcity to a point, then changes to computer precision search.
        """
        printc("GENERATING SPARSE DICT", 'cyan')
        global sparse_dict
        
        #printc("Building approximate sparse reservoirs for faster optimization ...",'fail')
        #for connectivity in np.logspace(0, -5, precision):
        #    sparse_dict[connectivity] = csc_matrix((self.accept < connectivity ) * self.reservoir_pre_weights)
        #self.sparse_keys_ = np.array(sorted(sparse_dict))
        self.number_of_preloaded_sparse_sets = len(reservoir_seeds)
        sparse_dict = GlobalSparseLibrary(precision = precision)
        for random_seed in reservoir_seeds:
            printc("generated sparse reservoir library for random seed " + str(random_seed), 'cyan')
            sparse_dict.addBook(random_seed)

    def get_approx_preRes(self, connectivity_threshold, i):
        """
        You can use the matrix returned instead of...
        """
        val = sparse_dict.get_approx_preRes(connectivity_threshold, index = i)
        return val

    def get_approx_preRes_old(self, connectivity_threshold, i):
        """
        You can use the matrix returned instead of...
        """
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  sparse_dict[key_].copy()
        return val

    def gen_in_weights(self):

        random_state = torch.Generator().manual_seed(self.seed_)

        n, m = self.n_nodes_, self.n_inputs_

        #at the moment all input weight matrices use uniform bias.
        self.uniform_bias = torch.FloatTensor(n, 1).uniform_(-1, 1, generator = random_state)

        #weights
        if self.input_weight_type_ == "uniform":
            
            self.in_weights_uniform = torch.FloatTensor(n, m).uniform_(-1, 1, generator = random_state)
            
            #add bias
            self.in_weights = torch.hstack((self.uniform_bias, self.in_weights_uniform))

        elif self.input_weight_type_ == "exponential":
            #printc("BUILDING SIGN_", 'fail')
            sign1 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))
            sign2 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))

            self.sign_dual = (sign1, sign2)
            self.sign = np.concatenate((sign1, sign2), axis = 1)

        #regularization
        self.noise_z = torch.normal(0, 1, size = (n, m + 1), generator = random_state)


class EchoStateNetworkCV:
    """A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors.

    Searches optimal solution within the provided bounds.

    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    model : class: {EchoStateNetwork, SimpleCycleReservoir}
            Model class to optimize
    subsequence_length : int
        Number of samples in one cross-validation sample
    eps : float
        The number specifying the maximum amount of change in parameters before considering convergence
    initial_samples : int
        The number of random samples to explore the  before starting optimization
    validate_fraction : float
        The fraction of the data that may be used as a validation set
    steps_ahead : int or None
        Number of steps to use in n-step ahead prediction for cross validation. `None` indicates prediction
        of all values in the validation array.
    max_iterations : int
        Maximim number of iterations in optimization
    batch_size : int
        Batch size of samples used by GPyOpt
    cv_samples : int
        Number of samples of the objective function to evaluate for a given parametrization of the ESN
    scoring_method : {'mse', 'rmse', 'tanh', 'nmse', 'nrmse', 'log', 'log-tanh', 'tanh-nrmse'}
        Evaluation metric that is used to guide optimization
    log_space : bool
        Optimize in log space or not (take the logarithm of the objective or not before modeling it in the GP)
    tanh_alpha : float
        Alpha coefficient used to scale the tanh error function: alpha * tanh{(1 / alpha) * mse}
    esn_burn_in : int
        Number of time steps to discard upon training a single Echo State Network
    acquisition_type : {'MPI', 'EI', 'LCB'}
        The type of acquisition function to use in Bayesian Optimization
    max_time : float
        Maximum number of seconds before quitting optimization
    n_jobs : int
        Maximum number of concurrent jobs
    esn_feedback : bool or None
        Build ESNs with feedback ('teacher forcing') if available
    update_interval : int (default 1)
        After how many acquisitions the GPModel should be updated
    verbose : bool
        Verbosity on or off
    plot : bool
        Show convergence plot at end of optimization
    target_score : float
        Quit when reaching this target score

    """

    def __init__(self, bounds, subsequence_length, model=EchoStateNetwork, eps=1e-8, initial_samples=50,
                 validate_fraction=0.2, steps_ahead=1, max_iterations=1000, batch_size=1, cv_samples=1,
                 scoring_method='nrmse', log_space=True, tanh_alpha=1., esn_burn_in=0, acquisition_type='LCB',
                 max_time=np.inf, n_jobs=1, random_seed=None, esn_feedback=None, update_interval=1, verbose=True,
                 plot=True, target_score=0., exp_weights = False, obs_index = None, target_index = None, noise = 0,
                 model_type = "random", activation_function = "tanh", input_weight_type = "uniform", 
                 Distance_matrix = None, n_res = 1, count = None, reservoir = None,
                 backprop = False, interactive = False
                 ):
        
        #############################new additions
        #real-time plotting
        self.errorz, self.errorz_step = [], [] #<-- consider re-naming
        self.interactive = interactive
        
        # backpropogation, real deep learning stuff!
        self.backprop = backprop
        #################################

        # Bookkeeping
        self.bounds = OrderedDict(bounds)
        self.free_parameters = []
        self.fixed_parameters = []
        self.n_res = n_res
        self.parameters = list(self.bounds.keys())

        if esn_feedback:
            printc("FEEDBACK " + str(esn_feedback), 'warning')

        self.count_ = count

        # Store settings
        self.model = model
        self.subsequence_length = subsequence_length
        self.eps = eps
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.steps_ahead = steps_ahead
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.cv_samples = cv_samples
        self.scoring_method = scoring_method
        self.log_space = log_space
        self.alpha = tanh_alpha
        self.esn_burn_in = torch.tensor(esn_burn_in, dtype=torch.int32).item()
        self.acquisition_type = acquisition_type
        self.max_time = max_time
        self.n_jobs = n_jobs
        self.seed = random_seed
        self.feedback = esn_feedback
        self.update_interval = update_interval
        self.verbose = verbose
        self.plot = plot
        self.target_score = target_score

        #Hayden modifications: varying the architectures
        self.exp_weights = exp_weights
        self.obs_index = obs_index
        self.target_index = target_index
        self.noise = noise
        self.model_type = model_type

        self.Distance_matrix = Distance_matrix
        assert self.n_jobs > 0, "njobs must be greater than 0"

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.parameters)

        self.activation_function = activation_function
        self.input_weight_type = input_weight_type

        if self.seed != None and type(self.bounds["n_nodes"]) == int:
            self.reservoir_matrices = ReservoirBuildingBlocks(model_type = self.model_type, 
                                                              random_seed = self.seed,
                                                              n_nodes = self.bounds["n_nodes"],
                                                              input_weight_type = self.input_weight_type)

    def normalize_bounds(self, bounds):
        """Makes sure all bounds feeded into GPyOpt are scaled to the domain [0, 1],
        to aid interpretation of convergence plots.

        Scalings are saved in instance parameters.

        Parameters
        ----------
        bounds : dicts
            Contains dicts with boundary information

        Returns
        -------
        scaled_bounds, scalings, intercepts : tuple
            Contains scaled bounds (list of dicts in GPy style), the scaling applied (numpy array)
            and an intercept (numpy array) to transform values back to their original domain
        """
        scaled_bounds = []
        scalings = []
        intercepts = []
        for name, domain in self.bounds.items():
            # Get any fixed parmeters
            if type(domain) == int or type(domain) == float:
                # Take note
                self.fixed_parameters.append(name)

            # Free parameters
            elif type(domain) == tuple:
                # Bookkeeping
                self.free_parameters.append(name)

                # Get scaling
                lower_bound = min(domain)
                upper_bound = max(domain)
                scale = upper_bound - lower_bound

                # Transform to [0, 1] domain
                scaled_bound = {'name': name, 'type': 'continuous', 'domain': (0., 1.)}

                # Store
                scaled_bounds.append(scaled_bound)
                scalings.append(scale)
                intercepts.append(lower_bound)
            else:
                raise ValueError("Domain bounds not understood")

        return scaled_bounds, np.array(scalings), np.array(intercepts)

    def denormalize_bounds(self, normalized_arguments):
        """Denormalize arguments to feed into model.

        Parameters
        ----------
        normalized_arguments : numpy array
            Contains arguments in same order as bounds

        Returns
        -------
        denormalized_arguments : 1-D numpy array
            Array with denormalized arguments

        """
        denormalized_bounds = (normalized_arguments.ravel() * self.bound_scalings) + self.bound_intercepts
        return denormalized_bounds

    def construct_arguments(self, x):
        """Constructs arguments for ESN input from input array.

        Does so by denormalizing and adding arguments not involved in optimization,
        like the random seed.

        Parameters
        ----------
        x : 1-D numpy array
            Array containing normalized parameter values

        Returns
        -------
        arguments : dict
            Arguments that can be fed into an ESN

        """

        # Denormalize free parameters
        denormalized_values = self.denormalize_bounds(x)
        arguments = dict(zip(self.free_parameters, denormalized_values))

        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value

        # Specific additions
        #arguments['random_seed'] = self.seed
        if 'regularization' in arguments:
            arguments['regularization'] = 10. ** arguments['regularization']  # Log scale correction

        log_vars = ["cyclic_res_w", "cyclic_input_w", "cyclic_bias"]

        for var in log_vars:
            if var in arguments:
                arguments[var] = 10. ** arguments[var]  # Log scale correction

        if 'connectivity' in arguments:
            arguments['connectivity'] = 10. ** arguments['connectivity']  # Log scale correction

        if 'llambda' in arguments:
            arguments['llambda'] = 10. ** arguments['llambda']  # Log scale correction

        if 'llambda2' in arguments:
            arguments['llambda2'] = 10. ** arguments['llambda2']  # Log scale correction

        if 'noise' in arguments:
            arguments['noise'] = 10. ** arguments['noise']  # Log scale correction

        if 'n_nodes' in arguments:
            arguments['n_nodes'] = int(np.round(arguments['n_nodes']))  # Discretize

        if not self.feedback is None:
            arguments['feedback'] = self.feedback

        return arguments

    def validate_data(self, y, x=None, verbose=True):
        """Validates inputted data against errors in shape and common mistakes.

        Parameters
        ----------
        y : numpy array
            A y-array to be checked (should be 2-D with series in columns)
        x : numpy array or None
            Optional x-array to be checked (should have same number of rows as y)
        verbose: bool
            Toggle to flag printed messages about common shape issues

        Raises
        ------
        ValueError
            Throws ValueError when data is not in the correct format.

        """
        # Check dimensions
        if not y.ndim == 2:
            raise ValueError("y-array is not 2 dimensional")

        if verbose and y.shape[0] < y.shape[1]:
            print("Warning: y-array has more series (columns) than samples (rows). Check if this is correct")

        # Checks for x
        if not x is None:

            # Check dimensions
            if not x.ndim == 2:
                raise ValueError("x-array is not 2 dimensional")

            # Check shape equality
            if x.shape[0] != y.shape[0]:
                raise ValueError("y-array and x-array have different number of samples (rows)")

    def optimize(self, y, x=None, store_path=None):
        """Performs optimization (with cross-validation).

        Uses Bayesian Optimization with Gaussian Process priors to optimize ESN hyperparameters.

        Parameters
        ----------
        y : numpy array
            Column vector with target values (y-values)

        x : numpy array or None
            Optional array with input values (x-values)

        store_path : str or None
            Optional path where to store best found parameters to disk (in JSON)

        Returns
        -------
        best_arguments : numpy array
            The best parameters found during optimization

        """
        # Checks
        self.validate_data(y, x, self.verbose)

        #generate the input weights for reuse. This will later to upgraded so that we can test cominations of reservoirs and input weights for robustness.
        #however, with the random state the same network and input weight matrices were being constantly recreated, leading to computational redundancy.
        self.reservoir_matrices.n_inputs_ = (y.shape[1] - 1) if type(x) == type(None) else x.shape[1]
        self.reservoir_matrices.gen_in_weights()

        # Initialize new random state
        self.random_state = np.random.RandomState(self.seed + 2)

        # Temporarily store the data
        self.x = x.astype(np.float32) if x is not None else None
        self.y = y.astype(np.float32)

        # Inform user
        if self.verbose:
            print("Model initialization and exploration run...")

        printc("njobs" + str(self.n_jobs), 'fail')

        # Define objective
        objective = GPyOpt.core.task.SingleObjective(self.objective_sampler,
                                                     objective_name='ESN Objective',
                                                     batch_type='synchronous',
                                                     num_cores=self.n_jobs)

        # Set search space and constraints
        space = GPyOpt.core.task.space.Design_space(self.scaled_bounds, constraints = None)

        # Select model and acquisition
        acquisition_type = self.acquisition_type
        model = RobustGPModel(normalize_Y = True, log_space = self.log_space)

        # Set acquisition
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')
        SelectedAcquisition = GPyOpt.acquisitions.select_acquisition(acquisition_type)
        acquisition = SelectedAcquisition(model = model, space = space, optimizer = acquisition_optimizer)

        # Add Local Penalization
        # lp_acquisition = GPyOpt.acquisitions.LP.AcquisitionLP(model, space, acquisition_optimizer, acquisition,
        # transform='none')
        printc("initial samples" + str(self.initial_samples), 'fail')

        # Set initial design
        n = len(self.free_parameters)
        initial_parameters = pyDOE.lhs(n, self.initial_samples, 'm')  # Latin hypercube initialization

        # Pick evaluator
        if self.batch_size == 1:
            
            evaluator = GPyOpt.core.evaluators.sequential.Sequential(acquisition=acquisition,
                                                                     batch_size=self.batch_size)
            #evaluator = GPyOpt.core.evaluators.ThompsonBatch(acquisition=acquisition,
            #                                                         batch_size=self.batch_size)
        else:
            evaluator = GPyOpt.core.evaluators.RandomBatch(acquisition=acquisition,
                                                           batch_size=self.batch_size)
            #evaluator = GPyOpt.core.evaluators.ThompsonBatch(acquisition=acquisition,
            #                                                         batch_size=self.batch_size)
        # Show progress bar
        if self.verbose:
            printc("Starting optimization..." + ' \n', 'green')

        ###
        print("Hayden edit: space: " + str(space))
        print("Hayden edit: fixed_parameters: " + str(self.fixed_parameters))
        print("Hayden edit: free_parameters: "  + str(self.free_parameters))
        ###
        # Build optimizer
        self.optimizer = EchoStateBO(model=model, space=space, objective=objective,
                                     acquisition=acquisition, evaluator=evaluator,
                                     X_init=initial_parameters, 
                                     model_update_interval=self.update_interval)

        # Optimize
        self.iterations_taken = self.optimizer.run_target_optimization(target_score=self.target_score,
                                                                       eps=self.eps,
                                                                       max_iter=self.max_iterations,
                                                                       max_time=self.max_time,
                                                                       verbosity=self.verbose)

        # Inform user
        if self.verbose:
            printc('Done after ' + str(self.iterations_taken) + ' iterations.', 'green')

        # Purge temporary data references
        del self.x
        del self.y

        # Show convergence
        if not store_path is None:
            plot_path = store_path[-5] + '_convergence.png'
        else:
            plot_path = None

        if self.plot or not store_path is None:
            self.optimizer.plot_convergence(filename=plot_path)

        # Store in dict
        best_arguments = self.construct_arguments(self.optimizer.x_opt)

        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)

        # Return best parameters
        return best_arguments


    def objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None, random_seed=None):
        """Returns selected error metric on validation set.

        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network, in column vector shape: (n, 1).
        train_y : array
            Dependent variable of the training set
        validate_y : array
            Dependent variable of the validation set
        train_x : array or None
            Independent variable(s) of the training set
        validate_x : array or None
            Independent variable(s) of the validation set

        Returns
        -------
        score : float
            Score on provided validation set

        """
        # Get arguments
        arguments = self.construct_arguments(parameters)

        #print("running objective function")
        try:
            res_args  = {"reservoir" : self.reservoir_matrices}
            arguments = {**arguments, **res_args}
        except:
            print("failed to load")


        # Build network
        esn = self.model(**arguments, exponential = self.exp_weights, activation_function = self.activation_function,
                obs_idx = self.obs_index, resp_idx = self.target_index, plot = False, model_type = self.model_type,
                input_weight_type = self.input_weight_type, Distance_matrix = self.Distance_matrix,
                random_seed = random_seed) 

        #if random_seed:
        #    print("initializing reservoir")
        #    print(self.count_)
        #    self.reservoir = esn.return_reservoir()
        #    self.in_weights = esn.return_in_weights()
        
        # Train
        esn.train(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        #print("validate y", validate_y.shape)
        score = esn.test(x=validate_x, y=validate_y, scoring_method=self.scoring_method,
                                         alpha=self.alpha)#, steps_ahead=self.steps_ahead) #validate_y.shape[0]) 
                         #steps_ahead=self.steps_ahead, alpha=self.alpha)
        return score

    ### Hayden Edit
    def define_tr_val(self, inputs):
        """
        Get indices
        start_index = np.random.randint(viable_start, viable_stop)
        train_stop_index = start_index + train_length
        validate_stop_index = train_stop_index + validate_length

        # Get samples
        train_y = training_y[start_index: train_stop_index]
        validate_y = training_y[train_stop_index: validate_stop_index]

        if not training_x is None:
            train_x = training_x[start_index: train_stop_index]
            validate_x = training_x[train_stop_index: validate_stop_index]
        else:
            train_x = None
            validate_x = None
        """
        ####
        #print("Hayden edit: parameters: " + str(parameters))
        #print("Hayden edit: fixed parameters: " + str(self.fixed_parameters))
        #print("Hayden edit: free parameters: " + str(self.free_parameters))
        ####
        start_index, random_seed = inputs["start_index"], inputs["random_seed"]
        # Get indices
        #start_index = np.random.randint(self.viable_start, self.viable_stop)
        train_stop_index = start_index + self.train_length
        validate_stop_index = train_stop_index + self.validate_length

        # Get samples
        train_y = self.y[start_index: train_stop_index]
        validate_y = self.y[train_stop_index: validate_stop_index]

        if not self.x is None:
            train_x = self.x[start_index: train_stop_index]
            validate_x = self.x[train_stop_index: validate_stop_index]
        else:
            train_x = None
            validate_x = None
        #return(train_x, train_y, validate_x, validate_y)
        # Loop through series and score result
        #scores_ = []
        
        
        score_ = self.objective_function(self.parameters, train_y, validate_y, train_x, validate_x, random_seed=random_seed)
        #scores_.append(score_)
        #for n in range(self.n_series):
        #    score_ = self.objective_function(self.parameters, train_y[:, n].reshape(-1, 1),
        #                                           validate_y[:, n].reshape(-1, 1), train_x, validate_x)
        #    scores_.append(score_)
        #print(scores_)
        return([score_])#{random_seed : score_})
    ###
    def build_unq_dict_lst(self, list1, list2, key1 = "start_index", key2 = "random_seed"):
        dict_list = []
        #print("list1:",list1)
        #print("list2:",list2)
        for i in range(len(list1)):

            if type(list2) == type(None):
                dictt = {}
                dictt[key1] =  lst1[i]
                dictt[key2] =  None
                dict_list.append(dictt)
            else:
                for j in range(len(list2)):
                    dictt = {}
                    dictt[key1] =  list1[i]
                    dictt[key2] =  list2[j]
                    dict_list.append(dictt)
        #print(len(dict_list))
        #dict_lst = list({v[key1]: v for v in dict_list}.values())
        #print(dict_lst)
        return dict_list

    def objective_sampler(self, parameters):
        """Splits training set into train and validate sets, and computes multiple samples of the objective function.

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

        # Get data
        self.parameters = parameters
        training_y = self.y
        training_x = self.x

        # Get number of series
        self.n_series = training_y.shape[1]

        # Set viable sample range
        viable_start = self.esn_burn_in
        viable_stop = training_y.shape[0] - self.subsequence_length

        # Get sample lengths
        self.validate_length = np.round(self.subsequence_length * self.validate_fraction).astype(int)
        self.train_length = self.subsequence_length - self.validate_length

        # Score storage
        scores = np.zeros((self.cv_samples, self.n_res), dtype=np.float32)

        ###
        start_indices = np.random.randint(viable_start, viable_stop, size = self.cv_samples)
        
       
        random_seeds  = np.random.randint(0, 100000, size = self.n_res) if self.seed == None else [self.seed]
        #random_seeds  = None if self.seed == None else [self.seed]

        objective_inputs = self.build_unq_dict_lst(start_indices, random_seeds)
       
        # Get samples
        #if not self.count_:
        #    results = self.define_tr_val(objective_inputs[0])

        if (self.cv_samples * self.n_res) > 1:
            Pool = multiprocessing.Pool(self.cv_samples * self.n_res)

            #get the asynch object:
            results = list(zip(*Pool.map(self.define_tr_val, objective_inputs)))

            Pool.close()
            Pool.join()
        else:
            results = self.define_tr_val(objective_inputs[0])

        self.scores = np.array(results).reshape(-1,1)

        #.reshape(scores.shape)

        # He is perhaps overlooking something essential to LCB here, it's hard to tell.
        # Why do we only pass back a mean?
        # Pass back as a column vector (as required by GPyOpt)
        #mean_score = self.scores.mean()
        mean_score = self.scores.mean()
        log_mu = round(np.log10(mean_score),4)
        
        #score_ = mean_score #- 0.1 * std_score
        
        score_str = 'iter ' + str(self.count_) +': Score ' + str(round(mean_score, 4)) +', log(\u03BC):' + str(log_mu) 
        
        if scores.shape[0] > 1:
            
            std_score = np.log10(self.scores.std())

            score_str += ", \u03C3: " + str(round(std_score, 4))
        score_str += " seed " + str(random_seeds) + " n " + str(self.scores.shape[0])

        if not self.count_:
            self.count_ = 1
        else:
            self.count_ +=1
        if log_mu > -0.5:
            printc(score_str, "fail")
        elif log_mu  > -0.8:
            printc(score_str, "warning")
        elif log_mu > -1:
            print(score_str)
        elif log_mu > -1.5:
            printc(score_str, "blue")
        elif log_mu > -2 :
            printc(score_str, "cyan")
        elif log_mu > -2.5:
            printc(score_str, "green")
            



        # Return scores
        return mean_score #- 0.1 * std_score #score_ #self.scores.reshape(-1,1)#mean_score.reshape(-1, 1)
