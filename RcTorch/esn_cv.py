from collections import OrderedDict
import copy
from dataclasses import dataclass
import json
from math import ceil, fabs
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool as mp_Pool
import multiprocessing
import pylab as pl
from IPython import display

from .esn import *

from copy import deepcopy


from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def get_initial_points(dim, n_pts, device, dtype):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def update_state(state, Y_next, dtype):
    """#TODO"""
    #hayden lines
    #if Y_next.dim() == 0:
    #    Y_next = float(Y_next) #.unsqueeze(dim = 1)
    ####
    if max(Y_next) > state.best_value + 1e-3 * fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts",
    dtype = torch.float32,
    device = None

):
    """#TODO"""
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates, dtype=dtype).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

class SparseBooklet:
    """TODO

    Arguments: TODO
    """
    def __init__(self, book, keys):
        self.sparse_book = book
        self.sparse_keys_ = np.array(keys)

    def get_approx_preRes(self, connectivity_threshold):
        """TODO

        Arguments: TODO
        """
        #print("sparse_keys", self.sparse_keys_, "connectivity_threshold", connectivity_threshold   )
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  self.sparse_book[key_].clone()
        return val

class GlobalSparseLibrary:
    """TODO

    Arguments: TODO
    """

    def __init__(self, lb = -5, ub = 0, n_nodes = 1000, precision = None, 
                 flip_the_script = False):
        self.lb = lb
        self.ub = ub
        self.n_nodes_ = n_nodes
        self.library = {}
        self.book_indices = []
        self.precision = precision
        self.flip_the_script = flip_the_script
        self.device = "cpu"
        

    def addBook(self, random_seed):
        """
        Add a sparse reservoir set.
        """
        book = {}
        n = self.n_nodes_
        
        random_state = Generator(device = self.device).manual_seed(random_seed)

        accept = rand(n, n, generator = random_state, device = self.device) 
        reservoir_pre_weights = rand(n, n, generator = random_state, device = self.device) * 2 -1

        "for now we're going to avoid sparse matrices"
        for connectivity in np.logspace(self.ub, self.lb, self.precision): #, device = self.device):
            #book[connectivity] = csc_matrix((accept < connectivity ) * reservoir_pre_weights)
            
            book[connectivity] = (accept < connectivity ) * reservoir_pre_weights
        sparse_keys_ = sorted(book)

        self.library[random_seed] = SparseBooklet(book = book, keys = sparse_keys_)
        self.book_indices.append(random_seed)

    def getIndices(self):
        """TODO"""
        return self.book_indices

    def get_approx_preRes(self, connectivity_threshold, index = 0):
        """
        You can use the matrix returned instead of...
        """
        if self.flip_the_script:
            index = np.random.randint(len(self.book_indices))
        #print("index", index, "book indices", self.book_indices, "self.library", self.library)
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
    def __init__(self, model_type, input_weight_type, random_seed , n_nodes, n_inputs = None, Distance_matrix = None, sparse = False):
        #print("INITIALING RESERVOIR")

        #initialize attributes
        self.device = "cpu"
        self.sparse = sparse
        self.random_seed = random_seed
        self.tensorArgs = {"device" : self.device}
        
        self.input_weight_type_ = input_weight_type
        self.model_type_ = model_type
        self.n_inputs_ = n_inputs
        n = self.n_nodes_ = n_nodes
        self.state = zeros(1, self.n_nodes_, device = self.device)

        if self.sparse:
        
            if model_type == "random":
                self.gen_ran_res_params()
                self.gen_sparse_accept_dict()
                assert 1 ==0
        else:
            random_state = Generator(device = self.device).manual_seed(random_seed)
            self.accept = rand(n, n, generator = random_state, device = self.device) 
            self.reservoir_pre_weights = rand(n, n, generator = random_state, device = self.device) * 2 -1


    def gen_ran_res_params(self):
        """TODO"""

        gen = Generator(device = self.device).manual_seed(self.random_seed)
        n = self.n_nodes_
        self.accept = rand(n, n, **self.tensorArgs, generator = gen)
        self.reservoir_pre_weights = rand(n, n, **self.tensorArgs, generator = gen) * 2 - 1

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
        sparse_dict = GlobalSparseLibrary(precision = precision, n_nodes = self.n_nodes_)
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
        val =  np.array(sparse_dict[key_]).copy()
        return val

    def gen_in_weights(self):
        """TODO"""

        gen = Generator(device = self.device).manual_seed(self.random_seed)
        n, m = self.n_nodes_, self.n_inputs_
        in_w_shape_ = (n, m)
        print('m,n', m,n)

        #at the moment all input weight matrices use uniform bias.
        self.bias = rand( n, 1, generator = gen, device = self.device) * 2 - 1

        #weights
        if self.input_weight_type_ == "uniform":
            self.in_weights = rand((n,m), generator = gen, device = self.device)
            self.in_weights = self.in_weights * 2 - 1
            print('in_weights', self.in_weights.shape)

        elif self.input_weight_type_ == "exponential":
            printc("BUILDING SIGN_", 'fail')
            sign1 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))
            sign2 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))

            self.sign_dual = (sign1, sign2)
            self.sign = np.concatenate((sign1, sign2), axis = 1)

        #regularization
        self.feedback_weights = rand(n, 1, **self.tensorArgs, generator = gen) * 2 - 1

        #regularization
        self.noise_z = normal(0, 1, size = (n, m), **self.tensorArgs, generator = gen)


__all__ = ['EchoStateNetworkCV']

def execute_HRC(arguments, upper_error_limit = 10000):
    """TODO

    Arguments: TODO
    """

    #later call define_tr_val from within this function for speedup.


    #score, pred_ = self.define_tr_val() #just get the train and test...
    #assert 1 == 0, "made it" + str(parallel_arguments)
    cv_samples, parallel_arguments, parameters, windowsOS, id_ = arguments
    device = parallel_arguments["device"]
    reservoir = parallel_arguments["declaration_args"]["reservoir"]

    if windowsOS == True:
        #move specific arguments to the gpu.
        cv_samples_ = []
        for i, cv_sample in enumerate(cv_samples):
            if  cv_sample["tr_y"].device != device:
                if cv_sample["tr_x"]:
                     train_x, validate_x = cv_sample["tr_x"].to(device), cv_sample["val_x"].to(device)
                else:
                    train_x, validate_x = None, None
                train_y, validate_y  = cv_sample["tr_y"].to(device), cv_sample["val_y"].to(device)
            else:
                #consider not sending the cv_sample["x"] if it's a pure prediction.
                train_x, train_y = cv_sample["tr_x"], cv_sample["tr_y"]
                validate_x, validate_y = cv_sample["val_x"], cv_sample["val_y"]
            cv_samples_.append((train_x, train_y, validate_x, validate_y))

            #now move the input weights and the reservoir arguments to the gpu.
            #deepcopy()
            if reservoir != None:
                reservoir.in_weights = reservoir.in_weights.to(device)
                reservoir.accept = reservoir.accept.to(device)
                reservoir.reservoir_pre_weights = reservoir.reservoir_pre_weights.to(device)

        del parallel_arguments["declaration_args"]["reservoir"]
        RC = EchoStateNetwork(**parallel_arguments["declaration_args"], reservoir = reservoir, **parameters, id_ = id_)
    else:
        RC = EchoStateNetwork(**parallel_arguments["declaration_args"], **parameters, id_ = id_)
        cv_samples_ = cv_samples
        cv_samples_ = []
        for i, cv_sample in enumerate(cv_samples):
            train_x, validate_x = cv_sample["tr_x"], cv_sample["val_x"]
            train_y, validate_y  = cv_sample["tr_y"], cv_sample["val_y"]
            cv_samples_.append((train_x, train_y, validate_x, validate_y))

    for i, cv_sample in enumerate(cv_samples_):

        train_x, train_y, validate_x, validate_y = cv_sample
        
        RC.train(X = train_x, y = train_y, **parallel_arguments["train_args"])

        del train_x; del train_y;

        # Validation score
        score, pred_, id_ = RC.test(x=validate_x, y= validate_y, **parallel_arguments["test_args"])

        del validate_x;

        if id_ != 0:
            del validate_y; del pred_;

        if not i:
            score_ = score
        else:
            score_ += score

        #if device == torch.device('cuda'):
        #    torch.cuda.empty_cache()
    score = min(score, tensor(upper_error_limit, device = device))

    if torch.isnan(score):
        score_ = tensor(upper_error_limit, device = device)

    del RC;
    score_mu = score_/len(cv_samples)
    del cv_samples;
    if id_ == 0:
        return float(score_mu), {"pred": pred_.to("cpu"), "val_y" : validate_y.to("cpu")}, id_
    else:
        return float(score_mu), None, id_

class EchoStateNetworkCV:
    """A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors.

    Searches optimal solution within the provided bounds.

    The acquisition function currently implimented is Thompson Sampling.

    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    model : class: {EchoStateNetwork}
            Model class to optimize
    subsequence_length : int
        Number of samples in one cross-validation sample
    initial_samples : int
        The number of random samples to explore the  before starting optimization
    validate_fraction : float
        The fraction of the data that may be used as a validation set
    batch_size : int
        Batch size of samples used by GPyOpt
    cv_samples : int
        Number of samples of the objective function to evaluate for a given parametrization of the ESN
    scoring_method : {'mse', 'rmse', 'tanh', 'nmse', 'nrmse', 'log', 'log-tanh', 'tanh-nrmse'}
        Evaluation metric that is used to guide optimization
    esn_burn_in : int
        Number of time steps to discard upon training a single Echo State Network
    esn_feedback : bool or None
        Build ESNs with feedback ('teacher forcing') if available
    verbose : bool
        Verbosity on or off

    device : 
        Torch device (either 'cpu' or 'cuda')
    interactive : BOOL
        if true, make interactive python plots. Useful in a jupyter notebook.
    approximate reservoir:
        if true, builds approximate sparse reservoirs and (ie approximate connectivity not precise). 
        It likely slightly reduces the final result's score but greatly speeds up the algorithm.
    input_weight_type : string
        {"uniform"} is currently implimented. 
        #TODO: exponential and normal weights.
    activation function: nn.function
        so far I've only played with tanh. Would be worth investigating other fuctions as well.
    model_type: 
        right now it is unclear whether this means reservoir type or model type.
        likely that is unclear because I haven't implimented cyclic or exponential here. Food for thought.

    failure tolerance: int
        the number of times that the model can fail to improve before length is in increased in turbo algo.
    success_tolerance: int
        like the explanation above this needs work.
    length_min: int
        ?????

    learning_rate:
        if backprop is True, then the RC will train with gradient descent. In this case this is that learning rate.



    #### BOTorch Turbo bayesian optimization parameters
    success_tolerance:
    failure_tolerance:

    

    #################### NOT IMPLIMENTED YET IN TORCH version (Hayden Fork)
    obs_index = None, target_index = None,  Distance_matrix = None, n_res = 1, 
    self.obs_index = obs_index
     self.target_index = target_index



    #################### NOT IMPLIMENTED IN TORCH version (came from Reinier)

    #################### eps, aquisition type and njobs seem like good things to port over.
    steps_ahead : int or None
        Number of steps to use in n-step ahead prediction for cross validation. `None` indicates prediction
        of all values in the validation array.
    max_iterations : int
        Maximim number of iterations in optimization
    log_space : bool
        Optimize in log space or not (take the logarithm of the objective or not before modeling it in the GP)
        ####### NOT IMPLIMENTED IN TORCH
    tanh_alpha : float
        Alpha coefficient used to scale the tanh error function: alpha * tanh{(1 / alpha) * mse}
    max_time : float
        Maximum number of seconds before quitting optimization
    acquisition_type : {'MPI', 'EI', 'LCB'}
        The type of acquisition function to use in Bayesian Optimization
    eps : float
        The number specifying the maximum amount of change in parameters before considering convergence
    plot : bool
        Show convergence plot at end of optimization
    target_score : float
        Quit when reaching this target score
    n_jobs : int
        Maximum number of concurrent jobs
    ####################

    """

    def __init__(self, bounds, subsequence_length, model=EchoStateNetwork, initial_samples=50,
                 validate_fraction=0.2, steps_ahead=1, batch_size=1, cv_samples=1,
                 scoring_method='nrmse', esn_burn_in=0, random_seed=None, esn_feedback=None, 
                 verbose=True, model_type = "random", activation_function = nn.Tanh(), 
                 input_weight_type = "uniform", backprop = False, interactive = False, 
                 approximate_reservoir = False, failure_tolerance = 1, length_min = 2**(-9), 
                 device = None, learning_rate = 0.005, success_tolerance = 3, dtype = torch.float32,
                 windowsOS = False):
        
        #### uncompleted tasks:
        #1) upgrade to multiple acquisition functions.
        #self.acquisition_type = acquisition_type

        ######

        #multiprocessing.set_start_method('spawn')
        self.windowsOS = windowsOS
        if not self.windowsOS:
            try:
                multiprocessing.set_start_method('spawn')
            except:
                pass
        if not device:
            self.device = torch_device("cuda" if cuda_is_available() else "cpu")
        else:
            self.device = device
        if self.device == torch_device('cuda'):
            torch.cuda.empty_cache()
        self.dtype = dtype

        print("FEEDBACK:", esn_feedback, ", device:", device)

        #turbo variables
        self.backprop = backprop
        self.batch_size = batch_size
        self.failure_tolerance = failure_tolerance
        self.length_min = length_min
        self.success_tolerance = success_tolerance


        #interactive plotting for jupyter notebooks.
        self.interactive = interactive

        self.learning_rate = learning_rate
        self.approximate_reservoir = approximate_reservoir

        # Bookkeeping
        self.bounds = bounds
        self.parameters = OrderedDict(bounds) 
        self.errorz, self.errorz_step = [], []
        self.free_parameters = []
        self.fixed_parameters = []
        #self.n_res = n_res

        # Store settings
        self.model = model
        self.subsequence_length = subsequence_length
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.steps_ahead = steps_ahead
        self.batch_size = batch_size
        self.cv_samples = cv_samples 
        self.scoring_method = scoring_method
        self.esn_burn_in =  tensor(esn_burn_in, dtype=torch.int32).item()   #torch.adjustment required
        self.esn_feedback = esn_feedback

        self.seed = random_seed
        self.feedback = esn_feedback
        self.verbose = verbose

        #Hayden modifications: varying the architectures
        
        self.model_type = model_type

        #self.Distance_matrix = Distance_matrix

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.parameters)
        
        self.activation_function = activation_function
        self.input_weight_type = input_weight_type
        
        if "n_nodes" in self.bounds:
            if type(["n_nodes"]) != int and type(["n_nodes"]) != float: #self.seed != None and 
                self.reservoir_matrices = None
            else:
                self.reservoir_matrices = ReservoirBuildingBlocks(model_type = self.model_type, 
                                                              random_seed = self.seed,
                                                              n_nodes = self.bounds["n_nodes"],
                                                              input_weight_type = self.input_weight_type)
        else:
            assert 1 == 0, "You must enter n_nodes as an argument into bounds_dict. ie: '\{ n_nodes: 1000 \}'"
            
        self.iteration_durations = []

        

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
        
        non_fixed_params = []
        
        print(self.device)
        
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
                #scaled_bound = {'name': name, 'type': 'continuous', 'domain': (0., 1.)} #torch.adjustment required
                non_fixed_params.append(name)
                
                # Store
                #scaled_bounds.append(scaled_bound)
                scalings.append(scale)
                intercepts.append(lower_bound)
            else:
                raise ValueError("Domain bounds not understood")
        
        n_hyperparams = len(non_fixed_params)
        
        scaled_bounds = cat([zeros(1,n_hyperparams, device = self.device), 
                                   ones(1, n_hyperparams, device = self.device)], 0)
        return scaled_bounds, tensor(scalings, device = self.device), tensor(intercepts, device = self.device) #torch.adjustment required

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
        denormalized_bounds = (normalized_arguments * self.bound_scalings) + self.bound_intercepts
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
        arguments = dict(zip(self.free_parameters, denormalized_values.flatten()))

        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value

        # Specific additions
        #arguments['random_seed'] = self.seed
        if 'regularization' in arguments:
            arguments['regularization'] = 10. ** arguments['regularization']  # Log scale correction

        #log_vars = ["cyclic_res_w", "cyclic_input_w", "cyclic_bias"]

        log_vars = ['connectivity', 'llambda', 'llambda2', 'noise']

        for var in log_vars:
            if var in arguments:
                arguments[var] = 10. ** arguments[var]  # Log scale correction

        if 'n_nodes' in arguments:
            arguments['n_nodes'] = tensor(arguments['n_nodes'], dtype = torch.int32, device = self.device)  # Discretize #torch.adjustment required

        if not self.feedback is None:
            arguments['feedback'] = self.feedback
        
        for argument, val_tensor in arguments.items():
            
            try:
                arguments[argument] = arguments[argument].item()
            except:
                arguments[argument] = arguments[argument]
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
    
    def eval_objective(self, x):
        """This is a helper function we use to unnormalize and evaluate a point"""
        return self.HRC(x)#unnormalize(x, self.scaled_bounds))

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
        arguments = self.construct_arguments(self.range_bounds)

        # Build network 
        esn = self.model(**arguments, activation_f = self.activation_function,
                plot = False, model_type = self.model_type,
                input_weight_type = self.input_weight_type, already_normalized = already_normalized)
                #random_seed = self.random_seed) Distance_matrix = self.Distance_matrix)
                #bs_idx = self.obs_index, resp_idx = self.target_index, 

        # Train
        esn.train(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test2(x=validate_x, y=validate_y, scoring_method=self.scoring_method, 
                            steps_ahead=self.steps_ahead, alpha=self.alpha)

        return score

    ### Hayden Edit
    def define_tr_val(self, inputs):
        """TODO

        Arguments: TODO
        """
        ####
        #print("Hayden edit: parameters: " + str(parameters))
        #print("Hayden edit: fixed parameters: " + str(self.fixed_parameters))
        #print("Hayden edit: free parameters: " + str(self.free_parameters))
        ####
        
        start_index, random_seed = inputs["start_index"], inputs["random_seed"]
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

        return {"tr_x" : train_x, "tr_y": train_y, "val_x": validate_x, "val_y" : validate_y }
        
    def build_unq_dict_lst(self, lst1, lst2, key1 = "start_index", key2 = "random_seed"):
        """TODO

        Arguments: TODO
        """
        dict_lst = []
        for i in range(len(lst1)):
            for j in range(len(lst2)):
                dictt = {}
                dictt[key1] =  lst1[i]
                dictt[key2] =  lst2[j]
                dict_lst.append(dictt)
        return dict_lst

    def objective_sampler(self): #, parameters
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
        #self.parameters = parameters
        training_y = self.y
        training_x = self.x

        # Get number of series
        self.n_series = training_y.shape[1]

        # Set viable sample range
        viable_start = self.esn_burn_in
        viable_stop = training_y.shape[0] - self.subsequence_length

        # Get sample lengths
        self.validate_length = torch.round(tensor(self.subsequence_length * self.validate_fraction)).type(torch.int32)
        self.train_length = self.subsequence_length - self.validate_length

        # Score storage
        #scores = torch.zeros(self.cv_samples, self.n_res, dtype = torch.float32)

        ### TORCH
        start_indices = randint(low = viable_start, high = viable_stop, size = (self.cv_samples,))
        start_indices = [index_tensor.detach() for index_tensor in start_indices]
        
        if self.seed == None:
            random_seeds  = randint(0, 100000, size = (self.n_res,), generator = self.random_state) #device = self.device, 
        else:
            random_seeds = [self.seed]

        objective_inputs = self.build_unq_dict_lst(start_indices, random_seeds)

       
        """
        # Get samples
        if (self.cv_samples * self.n_res)  > 1:
            Pool = mp.Pool(self.cv_samples * self.n_res)

            #get the asynch object:
            results = list(zip(*mp.Pool.map(self.define_tr_val, objective_inputs)))

            mp.Pool.close()
            mp.Pool.join()
        else:
            results = self.define_tr_val(objective_inputs[0])

        self.scores = tensor(results).view(-1,1)

        #.reshape(scores.shape)

        mean_score = self.scores.mean()
        
        print('Score \u03BC:' + str(round(np.log10(mean_score),4)), ", \u03C3: ",  round(np.log10(self.scores.std()),4), "seed", random_seeds, "n", self.scores.shape[0])#, "scores", self.scores)#str(self.scores.std()),)
            # pars = self.construct_arguments(parameters)

        # Return scores
        return mean_score #.reshape(-1, 1)
        """
        return self.define_tr_val(objective_inputs[0])
    
    def my_loss_plot(self, ax, pred, start_loc, valid, steps_displated = 500):#, pred = pred):
        """TODO

        Arguments: TODO
        """
        pred_ = pred.cpu().numpy()

        ax.plot(range(len(valid)), valid, color = 'blue', label = "train")

        #ax.plot(valid, color = "green", label = "test", alpha = 0.4)
        for i in range(len(valid) - pred.shape[1]):
            if i % 2 == 0:
                ax.plot(range(i, pred_.shape[1]+ i + 1),
                         cat([valid[i], tensor(pred_[i,:])],0), color = "red", alpha = 0.3)
        #ax.set_xlim(start_loc, 2200)

        plt.legend()
    
    def train_plot_update(self, pred_, validate_y, steps_displayed, elastic_losses = None):
        """TODO

        Arguments: TODO
        """
        
        #plotting
        if self.interactive:
            pred_2plot = pred_.clone().detach().to("cpu")
            validate_y_2plot = validate_y.clone().detach().to("cpu")
            try:
                self.ax[1].clear()
                self.ax[0].clear()
            except:
                pass
            #    self.ax[1].clear()
            #    #self.ax[1].plot(validate_y_2plot, alpha = 0.4, color = "blue") #[:steps_displayed]
            #plot 1:
            #log_resid = torch.log((validate_y - pred_)**2)
            labels = "best value", "all samples"
            self.ax[0].plot(np.log(self.errorz_step), alpha = 0.5, color = "blue", label = labels[0] )
            #print(self.errorz)
            self.ax[0].plot(np.log(self.errorz), alpha = 0.2, color = "green", label = labels[1])
            self.ax[0].set_title("log error vs Bayes Opt. step")
            self.ax[0].set_ylabel(f"log({self.scoring_method})")
            self.ax[0].set_xlabel("Bayesian Optimization step")
            self.ax[0].legend()
            
            log2 = np.log(2)
            self.ax[1].axhline(np.log(self.state.length_max)/log2, color = 'green', label = 'max length')
            self.ax[1].set_title("TURBO state")
            self.ax[1].plot(np.log(self.length_progress)/log2, color = 'blue', label = 'current length')
            self.ax[1].axhline(np.log(self.state.length_min)/log2, color = 'red', label = 'target length')
            self.ax[1].legend()

            #self.ax[1].set_ylim(self.y.min().item() - 0.1, self.y.max().item() )         

            #plot 3:
            self.ax[2].clear()
            self.ax[2].plot(validate_y_2plot[:steps_displayed].to("cpu"), alpha = 0.5, color = "blue", label = "val set")
            self.ax[2].plot(pred_2plot[:steps_displayed], alpha = 0.3, color = "red", label = "latest pred")
            
            #ax[2].plot(pred_[:steps_displayed], alpha = 0.5, color = "red", label = "pred")
            self.ax[2].set_ylim(self.y.min().item() - 0.1, self.y.max().item() )
            self.ax[2].set_title("Most recent validation prediction vs validation set")
            self.ax[2].set_ylabel("y")
            self.ax[2].set_xlabel("time step")
            pl.legend()

            display.clear_output(wait=True) 
            display.display(pl.gcf()) 

    

    
    def HRC(self, parameters, backprop = False, plot_type = "error", *args): #Hayden's RC or Hillary lol
        """
        This version of the RC helper function

        Arguments:
            parameterization
            steps_ahead
        The probelem with this parallelized version is almost certainly the fact that when we send the parallel computation,
        we don't get our results in any particular order. So what should we do?
        The answer is quite obvious: send them in with id tags and return the id tags: 0, through .shape[0].
        The resort on the backend.
        """
        parameter_lst = []
        for i in range(parameters.shape[0]):
            parameter_lst.append(self.construct_arguments(parameters[i, :]))
        
        start = time.time()

        declaration_args = {'activation_f' : self.activation_function,
                            'backprop' : self.backprop,
                            #'model_type' : self.model_type,
                            #'input_weight_type' : self.input_weight_type, 
                            'approximate_reservoir' : self.approximate_reservoir,
                            "device" : self.device,
                            "reservoir" : self.reservoir_matrices 
                            }
        train_args = {"burn_in" : self.esn_burn_in, "learning_rate" : self.learning_rate}
        test_args = {"scoring_method" : self.scoring_method}

        parallel_arguments = {"declaration_args": declaration_args, #"RC" : RC,
                              "train_args": train_args,
                              "test_args" : test_args,
                              "device" : self.device
                              }

        #the algorithm is O(n) w.r.t. cv_samples.
        cv_samples = [self.objective_sampler() for i in range(self.cv_samples)]
        
        data_args = [(cv_samples, parallel_arguments, params, self.windowsOS, i) for i, params in enumerate(parameter_lst)]

        num_processes = parameters.shape[0]

        if self.batch_size > 1:
            Pool = mp_Pool(num_processes)

            #get the asynch object:
            results = Pool.map(execute_HRC, data_args)
            
            Pool.close()
            Pool.join()

            results = sorted(results, key=lambda x: x[2]) 

            results = [(result[0], result[1]) for result in results]
            scores, preds = list(zip(*results)) 
        else:
            results = execute_HRC(data_args[0])
            scores, preds, id_ = results
            scores, preds = [scores], [preds]

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        for i, score in enumerate(scores):
            #score = score.view(1,1)
            if not i:
                Scores_ = [score]
            else:
                Scores_.append(score)
            self.errorz.append(score)
            #self.errorz_step.append(min(self.errorz))
            self.length_progress.append(self.state.length)

            
        #if self.count % self.batch_size == 0:
        if self.interactive:
            self.train_plot_update(pred_ = preds[0]["pred"], validate_y = preds[0]["val_y"], 
                steps_displayed = preds[0]["pred"].shape[0]) #l2_prop  = self.l2_prop) #elastic_losses = RC.losses, 

        Scores_ = tensor(Scores_, dtype = torch.float32, device = self.device).unsqueeze(-1)

        #print('success_tolerance', self.state.success_counter)
        #print('failure_tolerance', self.state.failure_counter)
        #score_str = 'iter ' + str(self.count) +': Score ' + f'{error.type(torch.float32).mean():.4f}'# +', log(\u03BC):' + f'{log_mu:.4f}' 

        """
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
        """
        stop = time.time()
        self.iteration_durations.append(stop - start)
        self.count += 1

        return - Scores_
    
    
    
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
        

        if self.interactive:
            self.fig, self.ax = pl.subplots(1,3, figsize = (16,4))
        
        # Checks
        self.validate_data(y, x, self.verbose)
        
        # Initialize new random state
        if self.reservoir_matrices != None:
            self.reservoir_matrices.n_inputs_ = max(y.shape[1] - 1, 1) if type(x) == type(None) else x.shape[1]
            
            self.reservoir_matrices.gen_in_weights()

        self.random_state = Generator().manual_seed(self.seed + 2)

        # Temporarily store the data
        #self.x = x.type(torch.float32).to(self.device) if x is not None else torch.ones(*y.shape, device = self.device)
        #self.y = y.type(torch.float32).to(self.device)  
        if type(y) == np.ndarray:
             y = torch.tensor(y, device = self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        if y.device != self.device:
            y = y.to(self.device)

        try:
            #X.any() this will fail if it is not np.array or a tensor (ie if it is bad input or None.)
            if type(x.any()) == np.ndarray:
                x = tensor(x, device = self.device)
            if x.device != self.device:
                x = x.to(self.device)
            if len(x.shape) == 1:
                x = x.view(-1, 1)
                orig_x = x.clone().detach()
        except:
            if not x:
                x = ones(*y.shape, device = self.device)
                orig_x = x.clone().detach()
        else:
            assert 0==5, "your input must  be a tensor, np.array or None."

        self.x = x.type(torch.float32) if x is not None else None #torch.ones(*y.shape)
        self.y = y.type(torch.float32)                           

        # Inform user
        if self.verbose:
            print("Model initialization and exploration run...")
            
        if self.interactive:
            self.fig, self.ax = pl.subplots(1,3, figsize = (16,4))
            
        self.errorz, self.errorz_step, self.length_progress = [], [], []
        dim = len(self.free_parameters)
        self.state = TurboState(dim, length_min = self.length_min, 
                                batch_size=self.batch_size, success_tolerance = self.success_tolerance)
        
        self.count = 1

        X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)
        
        if len(X_init) > self.batch_size:
            for i in range(X_init.shape[0] // self.batch_size):
                print(i)
                X_batch = X_init[ (i*self.batch_size) : ((i+1)*self.batch_size), : ]
                Y_batch = self.eval_objective( X_batch ) 
                if not i:
                    X_turbo = X_batch
                    Y_turbo = Y_batch 
                else:
                    Y_turbo = cat((Y_turbo, Y_batch), dim=0)
                    X_turbo = cat((X_turbo, X_batch), dim=0)
        else:

            X_turbo = X_init
            Y_turbo = self.eval_objective( X_init)
            
            #X_turbo.share_memory()
            #Y_turbo = tensor(
            #    [self.eval_objective(x.view(1,-1)) for x in X_turbo], dtype=dtype, device=device).unsqueeze(-1)
        X_turbo = X_turbo.to(self.device)
        Y_turbo = Y_turbo.to(self.device)
        
        n_init = self.initial_samples

        #append the errorz to errorz_step
        self.errorz_step += [max(self.errorz)] * X_turbo.shape[0] #n_init
        
        self.count = 0
        # Run until TuRBO converges
        while not self.state.restart_triggered:  
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X_turbo, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Create a batch
            X_next = generate_batch(
                state=self.state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=self.batch_size,
                n_candidates=min(5000, max(2000, 200 * dim)),
                num_restarts=10,
                raw_samples=512,
                acqf="ts",
                device = self.device
            )
            X_next = X_next

            #assert 1 ==0, X_next

            #can be parallelized:
            Y_next = self.eval_objective( X_next) #tensor()#.unsqueeze(-1)
                                    #[self.eval_objective(x.view(1,-1)) for x in X_next],
            print('Y_next', Y_next)
            print("self.state", self.state)
            # Update state
            self.state = update_state(state=self.state, Y_next=Y_next, dtype = self.dtype)

            # Append data
            X_turbo = cat((X_turbo, X_next), dim=0)
            Y_turbo = cat((Y_turbo, Y_next), dim=0)
            

            # Print current status
            print( 
                f"{len(X_turbo)}) Best score: {max(Y_next).item():.4f},  TR length: {self.state.length:.2e}" + 
                f" length {self.state.length}"# Best value:.item() {state.best_value:.2e},
            )
            
            print( 
                f"TR length: {self.state.length:.2e}," +  f" min length {self.state.length_min:.2e}"
                # + Best value:.item() {state.best_value:.2e},
            )

            self.errorz_step += [min(self.errorz)] * self.batch_size

            assert len(self.errorz) == len(self.errorz_step), "err len: {}, err step: {}".format(len(self.errorz), len(self.errorz_step) )
            
                    
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        best_vals = X_turbo[torch.argmax(Y_turbo)]

        self.X_turbo, self.Y_turbo =  X_turbo, Y_turbo
        
        denormed_ = self.denormalize_bounds(best_vals)
        
        try:
            denormed_ = denormalize_bounds(best_vals)
        except:
            print("FAIL")

        #####Bad temporary code to change it back into a dictionary
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        log_vars = ['connectivity', 'llambda', 'llambda2', 'noise', 'regularization']
        for var in log_vars:
            if var in best_hyper_parameters:
                best_hyper_parameters[var] = 10. ** best_hyper_parameters[var] 
        
        display.clear_output() 
        display.display(pl.gcf()) 
        
        # Return best parameters
        return best_hyper_parameters #X_turbo, Y_turbo, state, best_vals, denormed_ #best_arguments