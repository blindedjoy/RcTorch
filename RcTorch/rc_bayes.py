from collections import OrderedDict
import copy
from dataclasses import dataclass
import json
from math import ceil, fabs
import numpy as np
from scipy.sparse import csr_matrix
import pylab as pl
from IPython import display
from .defs import *
from .rc import *
from copy import deepcopy

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from matplotlib.ticker import MaxNLocator

import types
import functools

import itertools
import ray
from time import sleep

def _check_y(y, tensor_args = {}):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    if type(y) == np.ndarray:
         y = torch.tensor(y, **tensor_args)
    elif y.device != tensor_args["device"]:
        y = y.to(tensor_args["device"])
    if len(y.shape) == 1:
        y = y.view(-1, 1)
    return y

def _check_x(X, y, tensor_args = {}, supervised = False):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    if X is None:
        if supervised:
            X = torch.ones((y.shape[0],1), **tensor_args) #*y.shape,
        else:
            X = torch.linspace(0, 1, steps = y.shape[0], **tensor_args)
    elif type(X) == np.ndarray:
        X = torch.tensor(X,  **tensor_args)
    
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    return X


# ideas from yesterday (trip related)
# 1. moving average RC
# 2. Penalize the time the algorithm takes to run

@dataclass
class TurboState:
    """
    Summary line. This is from BOTorch. The Turbo state is a stopping condition.

    Extended description of function.

    Parameters
    ----------
    dim : dtype
        description
    batch_size : int
        description
    length_min : int
        description
    length_max : int
        description
    failure_counter : dtype
        description
    success_counter : dtype
        description
    success_tolerance: dtype
        description
    best_value: dtype
           the best value we have seen so far
    restart_triggered: dtype
        has a restart been triggered? If yes BO will terminat
    Returns
    -------
    int
        Description of return value

    """
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10 # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = ceil(
            max([5.0 , float(self.dim) ]) #/ self.batch_size / self.batch_size
        )

def get_initial_points(dim, n_pts, device, dtype):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def update_state(state, Y_next):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    """Updates the turbo state (checks for stopping condition)
    Essentially this checks our TURBO stopping condition.
    
    Arguments:
        state:  the Turbo state
        Y_next: the most recent error return by the objective function

    """
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
    model,
    X,  
    Y,
    batch_size,
    n_candidates=None,
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts",
    dtype = torch.float32,
    device = None
):
    """
    generate a batch for the Bayesian Optimization

    Extended description of function.

    Parameters
    ----------
    state : dtype
        the TURBO state (a stopping metric)
    model : dtype
        the GP (Gaussian Process) BOTorch model
    X : pytorch tensor
        points evaluated (a vector of hyper-parameter values fed to the objective function) # Evaluated points on the domain [0, 1]^d in original example, not ours.
    Y : pytorch tensor
        Function values
    n_candidates : 
        Number of candidates for Thompson sampling
    num_restarts : dtype
        description
    raw_samples : dtype
        describe
    acqf : 
        acquisition function (thompson sampling is preferred)

    Returns
    -------
    X_next : pytorch tensor
        The next set of normalized hyper-parameter values to try

    """
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        drawn = False
        while not drawn:

            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates, dtype=dtype).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=dtype, device=device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask        
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            try:
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():  
                    X_next = thompson_sampling(X_cand, num_samples=batch_size)
                    drawn = True
            except:
                X_next = torch.rand_like(X_cand[0].reshape(1,-1))

        


    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return X_next

#@ray.remote
class ReservoirBuildingBlocks:
    """ An object that allows us to save reservoir components (independent of hyper-parameters) for faster optimization.


    Parameters
    ----------
        model_type: either random, cyclic or delay line
        input_weight_type: exponential or uniform
        random_seed: the random seed to set the reservoir
        n_nodes: the nodes of the network
        n_inputs: the number of observers in the case of a block experiment, the size of the output in the case of a pure prediction where teacher forcing is used.
    

    """
    def __init__(self, model_type, input_weight_type, random_seed , n_nodes, n_inputs = None, 
                       Distance_matrix = None, sparse = False, device = None, reservoir_weight_dist = None):

        #initialize attributes
        self.device = device
        self.sparse = sparse
        self.random_seed = random_seed
        self.tensorArgs = {"device" : self.device}
        
        self.input_weight_type_ = input_weight_type
        self.model_type_ = model_type
        self.n_inputs_ = n_inputs
        n = self.n_nodes_ = n_nodes
        self.state = torch.zeros(1, self.n_nodes_, device = self.device)
        self.reservoir_weight_dist = reservoir_weight_dist

        if self.sparse:
            if model_type == "random":
                self.gen_ran_res_params()
                self.gen_sparse_accept_dict()
                assert 1 == 0
        else:
            gen = torch.Generator(device = self.device).manual_seed(self.random_seed)
            if reservoir_weight_dist == "uniform":
                self.reservoir_pre_weights = torch.rand(n, n, generator = gen, device = self.device) * 2 -1
            elif reservoir_weight_dist == "normal":
                shape_tuple = (self.n_nodes, self.n_nodes)
                ones_tensor, zeros_tensor = torch.ones(shape_tuple, **self.dev), torch.zeros(shape_tuple, **self.dev)
                self.weights = torch.normal(mean = ones_tensor, std = zeros_tensor) * self.sigma + self.mu

                self.weights *= (torch.rand(n, n, generator = random_state, device = self.device) < 0) * -1 
            else:
                assert False, f"{self.reservoir_weight_dist} reservoir_weight_distribution not yet implimented"
                

            random_state = torch.Generator(device = self.device).manual_seed(random_seed)
            self.accept = torch.rand(n, n, generator = random_state, device = self.device) 
            


    def gen_ran_res_params(self):
        """Generates the matrices required for generating reservoir weights"""
        gen = torch.Generator(device = self.device).manual_seed(self.random_seed)
        n = self.n_nodes_
        self.accept = torch.rand(n, n, **self.tensorArgs, generator = gen)
        self.reservoir_pre_weights = torch.rand(n, n, **self.tensorArgs, generator = gen) * 2 - 1

    def gen_sparse_accept_dict(self, reservoir_seeds = [123, 999], precision = 1000):
        """
        #TODO description

        Parameters
        ----------
        reservoir_seeds:
            preloaded reservoirs to generate (random seeds will uniquely create different reservoirs)
            precision: how precisely do you want to approximate connectivity in log space? 
        """
        printc("GENERATING SPARSE DICT", 'cyan')
        global sparse_dict
        
        #printc("Building approximate sparse reservoirs for faster optimization ...",'fail')
        #for connectivity in np.logspace(0, -5, precision):
        #    sparse_dict[connectivity] = csc_matrix((self.accept < connectivity ) * self.reservoir_pre_weights)
        #self.sparse_keys_ = np.array(sorted(sparse_dict))
        self.number_of_preloaded_sparse_sets = len(reservoir_seeds)
        sparse_dict = GlobalSparseLibrary(self.device, precision = precision, n_nodes = self.n_nodes_)
        for random_seed in reservoir_seeds:
            printc("generated sparse reservoir library for random seed " + str(random_seed), 'cyan')
            sparse_dict.addBook(random_seed)

    def get_approx_preRes(self, connectivity_threshold, i):
        """
        You can use the matrix returned instead of...
        TODO doctstring

        """
        val = sparse_dict.get_approx_preRes(connectivity_threshold, index = i)
        return val

    def get_approx_preRes_old(self, connectivity_threshold, i):
        """
        You can use the matrix returned instead of...

        Parameters
        ----------
        inputs : dict
            contains start_index and random_seed. 
                    (+) start index determines where to start the cross-validated sample
                    (+) random seed defines the reservoir's random seed.
        Returns
        ----------
        dict of new training and validation sets
        """
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  np.array(sparse_dict[key_]).copy()
        return val

    def gen_in_weights(self):
        """ Generates the reservoir input weight matrix
        
        This method assigns the reservoir input weights for later use downstream by the RcNetwork class.

        Parameters
        ----------
        inputs : None
        Returns: None
        ----------
        
        """

        gen = torch.Generator(device = self.device).manual_seed(self.random_seed)

        with torch.no_grad():
            n, m = self.n_nodes_, self.n_inputs_
            in_w_shape_ = (n, m)

            #at the moment all input weight matrices use uniform bias.
            self.bias = torch.rand( n, 1, generator = gen, device = self.device) * 2 - 1

            #weights
            if self.input_weight_type_ == "uniform":
                self.in_weights = torch.rand((n,m), generator = gen, device = self.device)
                self.in_weights = self.in_weights * 2 - 1

            elif self.input_weight_type_ == "exponential":
                printc("BUILDING SIGN_", 'fail')
                sign1 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))
                sign2 = random_state.choice([-1, 1], size= (in_w_shape_[0], in_w_shape_[1]//2))

                self.sign_dual = (sign1, sign2)
                self.sign = np.concatenate((sign1, sign2), axis = 1)

            #regularization
            self.feedback_weights = torch.rand(n, m, **self.tensorArgs, generator = gen) * 2 - 1

            #regularization
            self.noise_z = torch.normal(0, 1, size = (n, m), **self.tensorArgs, generator = gen)


__all__ = ['RcBayesOpt']

def process_score(score__, upper_error_limit = 1000000, device = None):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    if torch.isnan(score__):
        score__ = torch.tensor(upper_error_limit, device = device, requires_grad  = False, dtype = torch.float32)
    else:
        score__ = min(score__, torch.tensor(upper_error_limit, device = device, requires_grad = False, dtype = torch.float32))
    return score__

def combine_score(tr_score, val_score, tr_score_prop, log_score):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    tr_score = tr_score.type(torch.float32)
    val_score = val_score.type(torch.float32)
    if log_score:
        tr_score = torch.log10(tr_score)
        val_score = torch.log10(val_score)
        return tr_score * tr_score_prop + val_score * (1- tr_score_prop)
    else:
        return torch.log10(tr_score * tr_score_prop + val_score * (1- tr_score_prop))

if __name__ == "__main__":
    CUDAA = torch.cuda.is_available()
    if CUDAA:
        print("cuda is available")
        n_gpus = 0.1
    else:
        print("cuda is not available")
        n_gpus = 0
else:
    CUDAA = torch.cuda.is_available()
    if CUDAA:
        n_gpus = 0.1
    else:
        n_gpus = 0

@ray.remote(num_gpus=n_gpus, max_calls=1)
def execute_objective(parallel_arguments, parameters, X_turbo_spec, trust_region_id):#arguments):
    """ Function at the heart of RCTorch, train a network on multiple rounds of cross-validated train/test info, then return the average error.
    This method also deals with dispatching mutliple series to the objective function if there are multiple, and aggregates the returned scores
        by averaging.

    Parameters
    ----------
        arguments: a list of arguments that have been put into dictionary form for multiprocessing convenience
        upper_error_limit: brutal error clipping upper bound, 'nan' error function returns the maximum upper limit as well to discourage
                the algorithm from searching in that part of the parameter space.

    Returns
    -------
        tuple of score (float), the prediction and validation sets for plotting (optional), and the job id

    (we need the job id to resort and relate X and y for BO Opt which have been scrambled by multiprocessing)
    """

    #parallel_arguments
    # device, delcaration_args, backprop_args, cv_args, train_args, test_args
    device = parallel_arguments["device"]
    declaration_args = parallel_arguments["declaration_args"]
    backprop_args = parallel_arguments["backprop_args"]
    cv_args =  parallel_arguments["cv_args"]
    log_score, tr_score_prop = cv_args["log_score"],  cv_args["tr_score_prop"]

    total_score = 0

    RC = RcNetwork(**declaration_args, **parameters, id_ = 1)
    train_args = parallel_arguments["train_args"]
    test_args = parallel_arguments["test_args"]
    fit_inputs = parallel_arguments["fit_inputs"]
    val_inputs = parallel_arguments["val_inputs"]
    n_cv_samples = len(fit_inputs)

    ode = train_args["ODE_order"]

    for i, fit_input in enumerate(fit_inputs):

        cv_sample_score = 0

        val_input = val_inputs[i]

        if ode:
            results = RC.fit(**fit_input, train_score = True, **train_args)
            train_scores = results["scores"]
            val_scores, pred_, id_ = RC.test(**val_input, **test_args)

            for i, train_score in enumerate(train_scores):
                
                train_scores[i] = process_score(train_score, device = device)
                val_score = process_score(val_scores[i], device = device)# / divis
                round_score = combine_score(train_score, val_score, tr_score_prop, log_score)  
                cv_sample_score += round_score 
            total_score += cv_sample_score

            
        else:
            _ = RC.fit(**fit_input, **train_args)
            
            val_score, pred_, id_ = RC.test(**val_input, **test_args)
            val_score = cv_sample_score = process_score(val_score)
            val_scores = [torch.log10(val_score)]

        total_score += cv_sample_score

    total_score = total_score / (n_cv_samples * len(val_scores))
    
    common_args = {"X_turbo_spec" : X_turbo_spec, "trust_region_id" : trust_region_id} #, "job_id" : job_id
    if ode:
        best_batch_score = np.min(results["scores"])
        best_idx = np.argmin(results["scores"])
        return float(total_score), {"pred": results["ys"][best_idx].to("cpu"), 
                                    "val_y" : results["ydots"][best_idx], 
                                    "score" : best_batch_score, 
                                    **common_args}
    else:
        return float(total_score), {"pred": pred_, 
                                    "val_y" : val_input["y"], 
                                    "score" : val_score, 
                                    **common_args}

#@ray.remote( max_calls=1)
def eval_objective_remote(parallel_args_id, parameters, dtype = None, device = None, plot_type = "error",  *args):
        """
        This version of the RC helper function

        Parameters
        -------
        parameters: torch.tensor
            a torch.tensor of the hyper-paramters drawn from the BO_step at time t

        plot_type

        Returns
        -------

        """
        parameter_lst, X_turbo_batch, trust_region_ids = parameters
        
        num_processes = len(parameter_lst)
        #job_ids = [id(parameter_lst[i]) for i in range(num_processes)]

        #What is returned by execute_objective:

        #     return float(total_score), {"pred": results["ys"][best_idx].to("cpu"), "val_y" : results["ydots"][best_idx], "score" : best_batch_score, "id" : id_}
        # else:
        #     return float(total_score), {"pred": pred_, "val_y" : val_input["y"], "score" : val_score, "id" : id_}

        #parallel_arguments, parameters, X_turbo_spec
        results = ray.get([execute_objective.remote(parallel_args_id, params, X_turbo_batch[i], trust_region_ids[i]) for i, params in enumerate(parameter_lst)])

        #the old way of sorting orgainizing to avoid problems with parallel job asynch, new method is id internal.
        #results = sorted(results, key=lambda x: x[1]["trust_region_id"]) 
        
        scores, result_dicts = list(zip(*results)) 

        k = best_score_index = np.argmin(scores)

        X_turbo_specs = torch.vstack([result_dict["X_turbo_spec"] for result_dict in result_dicts])

        trust_region_ids = [result_dict["trust_region_id"] for result_dict in result_dicts]

        batch_dict = {"pred" : result_dicts[k]["pred"], 
                      "y" : result_dicts[k]["val_y"], 
                      "trust_region_ids": trust_region_ids, 
                      "best_score" : min(scores)}
       
        for i, score in enumerate(scores):
            if not i:
                Scores_ = [score]
            else:
                Scores_.append(score)
        
        Scores_ = torch.tensor(Scores_, dtype = dtype, device = device, requires_grad = False).unsqueeze(-1)

        return X_turbo_specs, -Scores_, batch_dict

def if_split(tensor, start_index, train_stop_index, validate_stop_index):
    """
    TODO doctstring
    """
    if not tensor is None:
        train_tensor = tensor[start_index: train_stop_index]
        validate_tensor = tensor[train_stop_index: validate_stop_index]
    else:
        train_tensor, validate_tensor = None, None
    return train_tensor, validate_tensor

class RcBayesOpt:
    """A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors.

    Searches optimal solution within the provided bounds.

    The most important argument is the :attr:`bounds` argument which defines the search space for the various
    hyper-parameters. An example of this argument is :

    .. code-block:: python

       bounds_dict = { "connectivity" : (0,1),
           "spectral_radius" : (0.6, 2),
           "n_nodes" : (1, 353.1),
           "log_regularization" : (-3, 3),
           "leaking_rate" : (0, 1),
           "input_connectivity" : (0, 1),
           "feedback_connectivity" : (0, 1),
           "bias": (0, 1),
           }
    .. note::
        You can search in log-space for any hyper-parameter by including 'log_' in the string. For example,
        if we instead wanted to search for the connectivity between 0.01 and 0.1 we could modify the connectivity argument in the bounds dict above
        to 'log_connectivity : (-2, -1)'.
    .. warning::
        The only acquisition function which is currently implimented is Thompson Sampling.

    

        

    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    model : class: {RcNetwork}
            Model class to optimize
    subsequence_length : int
        Number of samples in one cross-validation sample
    initial_samples : int
        The number of random samples to explore the  before starting optimization
    validate_fraction : float
        The fraction of the data that may be used as a validation set
    batch_size : int
        Batch size of samples used by BoTorch
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
    device : string or torch device                                 
        Torch device (either 'cpu' or 'cuda')
    interactive : bool
        if true, make interactive python plots. Useful in a jupyter notebook.
    approximate reservoir: bool
        if true, builds approximate sparse reservoirs and (ie approximate connectivity not precise). 
        It likely slightly reduces the final result's score but greatly speeds up the algorithm. #SPARCITY NOT IMPLIMENTED IN RCTORCH
    input_weight_type : string
        {"uniform"} is currently implimented. 
        #TODO: exponential and normal weights.
    activation function: nn.function
        The activation function used in the reservoir
    model_type: str
            #TODO
        right now it is unclear whether this means reservoir type or model type.
        likely that is unclear because I haven't implimented cyclic or exponential here. #TODO impliment uniform and expo weights
    failure tolerance: int
        the number of times that the model can fail to improve before length is in increased in turbo algo.
    success_tolerance: int
        like the explanation above this needs work.
    length_min: int
        The stopping condition. If the turbo_state's length falls below length_min then the algorithm will terminate.
    learning_rate: float
        if backprop is True, then the RC will train with gradient descent. In this case this is that learning rate.
    success_tolerance:
        #TODO description
    failure_tolerance:
        #TOD description
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

    """
    ####################

    #### uncompleted tasks:
    #1) upgrade to multiple acquisition functions.
    #self.acquisition_type = acquisition_type
    ######
    #################### NOT IMPLIMENTED YET IN TORCH version (Hayden Fork)
    #obs_index = None, target_index = None,  Distance_matrix = None, n_res = 1, 
    #self.obs_index = obs_index
    # self.target_index = target_index
    #################### NOT IMPLIMENTED IN TORCH version (came from Reinier)
    #################### eps, aquisition type and njobs seem like good things to port over.

    def __init__(self, bounds, subsequence_prop = 0.8,  model=RcNetwork, initial_samples=50, #subsequence_length,
                 validate_fraction=0.5, steps_ahead=None, turbo_batch_size=1, cv_samples=1, n_jobs = 1,
                 scoring_method='nrmse', esn_burn_in=0, random_seed=None, feedback=None, 
                 verbose=True, model_type = "random", activation_function = 'sigmoid', #nn.Tanh(), 
                 output_activation = "identity",
                 input_weight_type = "uniform", interactive = True, 
                 approximate_reservoir = False, length_min = 2**(-9), 
                 device = None, success_tolerance = 3, dtype = torch.float32,
                 windowsOS = False, track_in_grad = False, patience = 400, ODE_order = None,
                 dt = None, log_score =  False, n_inputs = None, n_outputs = None,
                 reservoir_weight_dist = "uniform", feedback_weight_dist = "uniform", input_weight_dist = "uniform",
                 solve_sample_prop = 1
                 ):
        # assert isinstance(n_inputs, int), "you must enter n_inputs. This is the number of input time series (int)"
        # assert isinstance(n_outputs, int), "you must enter n_outputs. This is the number of output time series (int)"
        
        #self.n_res = n_res

        #assign attributes to self
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)

        self.batch_size = self.n_jobs

        self.esn_burn_in = torch.tensor(esn_burn_in, dtype=torch.int32).item()

        self.parameters = OrderedDict(bounds) 

        
        self._errorz, self._errorz_step = [], []
        self.free_parameters = []
        self.fixed_parameters = []

        if not device:
            self.device = torch_device("cuda" if cuda_is_available() else "cpu")
        else:
            self.device = device
        if self.device == torch_device('cuda'):
            torch.cuda.empty_cache()
        
        #self.Distance_matrix = Distance_matrix

        self._check_bounds(self.parameters)

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self._normalize_bounds(self.parameters)
        
        if "n_nodes" not in self.bounds:
            assert 1 == 0, "You must enter n_nodes as an argument into bounds_dict. ie: '\{ n_nodes: 1000 \}'"
            
        self.iteration_durations = []


    def _check_bounds(self, bounds):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """

        prob = (0,1)
        log_prob = (None, np.log10(1))

        bound_limits = {"connectivity" : prob, 
                        "input_connectivity" : prob, 
                        "feedback_connectivity" : prob, 
                        "leaking_rate" : prob,
                        "n_nodes" : (1, 10000000000),
                        "log_connectivity" : log_prob,
                        "log_input_connectivity" : log_prob,
                        "log_feedback_connectivity" : log_prob,
                        "log_leaking_rate" : log_prob
                        }

        for var in bound_limits:
            if var in list(bounds.keys()):
                llim, ulim = bound_limits[var]

                if type(bounds[var]) in [tuple, list]:
                    
                    if llim is not None:
                        if bounds[var][0] < llim:
                            assert False, f'{var} limit is illegal, the bound cannot be lower than {llim}'
                    if ulim is not None:
                        if bounds[var][1] > ulim:
                            assert False, f'{var} limit is illegal, the bound cannot be greater than {ulim}'
                
                elif type(bounds[var]) in [int, float]:
                    if llim is not None:
                        if bounds[var] < llim:
                            assert False, f'{var} limit is illegal, the bound cannot be lower than {llim}'
                    if ulim is not None:
                        if bounds[var] > ulim:
                            assert False, f'{var} limit is illegal, the bound cannot be greater than {ulim}'
                else:
                    assert False, f"bad bounds type {var} {type(var_bounds)}"



        

    def _normalize_bounds(self, bounds):
        """Makes sure all bounds feeded into BoTorch are scaled to the domain [0, 1],
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
        
        scaled_bounds = torch.cat([torch.zeros(1,n_hyperparams, device = self.device), 
                                   torch.ones(1, n_hyperparams, device = self.device)], 0)
        return scaled_bounds, torch.tensor(scalings, device = self.device, requires_grad = False), torch.tensor(intercepts, device = self.device, requires_grad = False) #torch.adjustment required

    def _denormalize_bounds(self, normalized_arguments):
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

    def _construct_arguments(self, x):
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
        denormalized_values = self._denormalize_bounds(x)

        arguments = dict(zip(self.free_parameters, denormalized_values.flatten()))

        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value

        user_var_list = list(arguments.keys())
        log_var_indices = ['log_' in str_ for str_ in user_var_list]
        user_variables = [(str_.split(sep = 'log_'))[-1] for str_ in user_var_list]

        self.log_vars = list(np.array(user_variables)[log_var_indices])

        for var in self.log_vars:
            arguments[var] = 10. ** arguments['log_' + var]
            del arguments['log_' + var]

        if 'n_nodes' in arguments:
            if type(arguments['n_nodes']) in [int, float]:
                arguments['n_nodes'] = torch.tensor(arguments['n_nodes'], dtype = torch.int32, device = self.device, requires_grad = False)  # Discretize #torch.adjustment required
            else:
                arguments['n_nodes'] = arguments['n_nodes'].type(dtype = torch.int32).to(self.device)

        if not self.feedback is None:
            arguments['feedback'] = self.feedback
        
        for argument, val_tensor in arguments.items():
            
            try:
                arguments[argument] = arguments[argument].item()
            except:
                arguments[argument] = arguments[argument]
        return arguments

    def _validate_data(self, y, x=None, verbose=True):
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
        if not self.ODE_order:
            if not y.ndim == 2:
                raise ValueError("y-array is not 2 dimensional, if ODE and you didn't provide y then x is one dim")

            if verbose and y.shape[0] < y.shape[1]:
                print("Warning: y-array has more series (columns) than samples (rows). Check if this is correct")

        # Checks for x
        if self.ODE_order and x is None:
            assert False
        if not x is None:

            # Check dimensions
            if not x.ndim == 2:
                raise ValueError("x-array is not 2 dimensional")

            # Check shape equality
            if x.shape[0] != y.shape[0]:
                raise ValueError("y-array and x-array have different number of samples (rows)")
    

    def _objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None, random_seed=None):
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
        arguments = self._construct_arguments(self.range_bounds)

        # Build network 
        esn = self.model(**arguments, activation_f = self.activation_function,
                plot = False, model_type = self.model_type,
                input_weight_type = self.input_weight_type, already_normalized = already_normalized)
                #random_seed = self.random_seed) Distance_matrix = self.Distance_matrix)
                #bs_idx = self.obs_index, resp_idx = self.target_index, 

        # Train
        esn.fit(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test2(x=validate_x, y=validate_y, scoring_method=self.scoring_method, 
                            steps_ahead=self.steps_ahead, alpha=self.alpha)

        return score

    def _define_tr_val(self, inputs):
        """Splits training sets (X, y) into train and validate sets, in order to later compute multiple samples of the objective function.

        Parameters
        ----------
        inputs : dict
            contains start_index and random_seed. 
                    (+) start index determines where to start the cross-validated sample
                    (+) random seed defines the reservoir's random seed.
        Returns
        ----------
        dict of new training and validation sets
        """
        
        start_index, random_seed = inputs["start_index"], inputs["random_seed"]
        train_stop_index = start_index + self.train_length
        validate_stop_index = train_stop_index + self.validate_length

        # Get samples
        if self.ODE_order:
            train_y = None
            val_y = None
        else:
            train_y = self.y[start_index: train_stop_index]
            val_y = self.y[train_stop_index: validate_stop_index]
            

        train_x, val_x = if_split(self.x, 
                                        start_index, 
                                        train_stop_index,
                                        validate_stop_index )

        ##################### beta arguments are currently silenced ###############

        # train_beta, val_beta = if_split(self.beta, 
        #                                      start_index, 
        #                                      train_stop_index, 
        #                                      validate_stop_index )

        ##################### beta arguments are currently silenced ###############

        fit_dict = {"X": train_x,
                    "y": train_y}

        val_dict = {"X": val_x,
                    "y": val_y}

        return (fit_dict, val_dict)
        
        
    def _build_unq_dict_lst(self, lst1, lst2, key1 = "start_index", key2 = "random_seed"):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        """This function builds a list of dictionairies with unique keys.

        Arguments: TODO
        TODO doctstring

        """
        dict_lst = []
        for i in range(len(lst1)):
            for j in range(len(lst2)):
                dictt = {}
                dictt[key1] =  lst1[i]
                dictt[key2] =  lst2[j]
                dict_lst.append(dictt)
        return dict_lst

    def _objective_sampler(self):
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
            Column vector with mean score(s), (as was required by GPyOpt)

        """
        # Get data
        #self.parameters = parameters
        training_y = self.y
        training_x = self.x

        
        # Set viable sample range
        if not self.ODE_order:
            viable_start = self.esn_burn_in
            # Get number of series
            self.n_series = training_y.shape[1]
            viable_stop = training_y.shape[0] - self.subsequence_length
        else:
            viable_start = 0
            # Get number of series
            self.n_series = training_x.shape[1]
            viable_stop = training_x.shape[0] - self.subsequence_length

        # Get sample lengths
        self.validate_length = torch.round(torch.tensor(self.subsequence_length * self.validate_fraction, requires_grad  = False)).type(torch.int32)
        self.train_length = self.subsequence_length - self.validate_length

        ### TORCH
        start_indices = torch.randint(low = viable_start, high = viable_stop, size = (self.cv_samples,))
        start_indices = [index_tensor.detach() for index_tensor in start_indices]
        
        if self.random_seed == None:
            random_seeds  = torch.randint(0, 100000, size = (self.n_res,), generator = self.random_state) #device = self.device, 
        else:
            random_seeds = [self.random_seed]

        objective_inputs = self._build_unq_dict_lst(start_indices, random_seeds)

        return self._define_tr_val(objective_inputs[0])
    
    def _my_loss_plot(self, ax, pred, start_loc, valid, steps_displated = 500):#, pred = pred):
        """

        Parameters
        ----------

        ax

        pred: ???
        ????

        start_loc: ???
        ????

        valid: ????
        ????

        steps displated: int
        
        Returns
        -------


        Arguments: TODO
        """
        pred_ = pred.cpu().numpy()

        ax.plot(range(len(valid)), valid, color = 'blue', label = "train")

        #ax.plot(valid, color = "green", label = "test", alpha = 0.4)
        for i in range(len(valid) - pred.shape[1]):
            if i % 2 == 0:
                ax.plot(range(i, pred_.shape[1]+ i + 1),
                         torch.cat([valid[i], torch.tensor(pred_[i,:])],0), color = "red", alpha = 0.3)
        #ax.set_xlim(start_loc, 2200)

        plt.legend()
    
    def _train_plot_update(self, pred_, validate_y, steps_displayed, elastic_losses = None, restart_triggered = False):
        """If you are running rctorch in a jupyter notebook then this function displays live plots so that you can watch training if 
        self.interactive = True.

        

        Parameters
        ----------
        pred_: torch.tensor
            the model's prediction
        validate_y: torch.tensor
            the validation set of the response
        steps_displayed: int
            the number of timesteps of pred_ and validate_y to show in the plot
        elastic_losses:

        """
        if self.interactive:
            display.clear_output(wait=True) 
            
            pred_2plot = pred_.detach().to("cpu")
            if not self.ODE_order:
                validate_y_2plot = validate_y.detach().to("cpu")
            try:
                self.ax[1].clear()
                self.ax[0].clear()
            except:
                pass

            labels = "best value", "all samples"

            # Plot 1: the training history of the bayesian optimization
            
            font_dict = legend_font_dict =  {"prop" : {'size': 12}}
            ticks_font_size = 14

            len_min, len_max = self.states[0].length_min, self.states[0].length_max

            plot = self.ax

            for i in range(self.n_trust_regions):
                if not i:
                    labels_ = labels
                    plot2_label = 'current length'
                else:
                    labels_ = None, None
                    plot2_label = None
                
                plot[0].plot(self._errorz_step[i], alpha = 0.5, color = "blue", label = labels_[0] )
                plot[0].plot(self._errorz[i], alpha = 0.2, color = "green", label = labels_[1])
                

                #plot 2: the turbo state
                plot[1].plot(np.log(self._length_progress[i])/self.log2, color = 'blue', label = plot2_label)

            if self.n_trust_regions > 1:
                pct_complete = (self.n_evals/self.max_evals) * 100
                plot[0].set_title(f'% complete: { pct_complete:.0f}, n_evals: {self.n_evals}')

            
            #self.ax[0].set_title("log(error) vs Thompson Sampling step")
            plot[0].set_ylabel(f"log({self.scoring_method})")
            plot[0].set_xlabel("BO step")
            plot[0].set_ylabel("Error")
            plot[0].legend(**font_dict)
            plot[0].set_ylim(min(self._errorz["all"])/2 , max(1.05, np.quantile(self._errorz["all"], 0.95)))
            plot[0].set_yscale("log")
            
            plot[0].xaxis.set_major_locator(MaxNLocator(integer=True))

            plot[1].axhline(np.log(len_max)/self.log2, color = 'green', label = 'max length')
            plot[1].set_title("TURBO state")
            plot[1].axhline(np.log(len_min)/self.log2, color = 'red', label = 'target length')
            plot[1].legend(**legend_font_dict)

            #plot 3 (most recent prediction)
            plot[2].clear()
            if self.ODE_order and pred_2plot.shape[1] == 2:
                plot[2].plot(pred_2plot[:,0], pred_2plot[:,1], alpha = 0.3, color = "red", label = "latest pred")
                plot[2].set_title("Phase space")
            else:
                if not self.ODE_order:
                    plot[2].plot(validate_y_2plot[:,0].to("cpu"), alpha = 0.5, color = "blue", label = "ground truth")
                plot[2].plot(pred_2plot[:,0], alpha = 0.3, color = "red", label = "RC")
                    
                if pred_2plot.shape[1] > 1:
                    if not self.ODE_order:
                        
                        plot[2].plot(validate_y_2plot[:,1:].to("cpu"), alpha = 0.5, color = "blue", label = None)
                        plot[2].set_ylim(self.y.min().item() - 0.1, self.y.max().item() )
                    
                    plot[2].plot(pred_2plot[:,1:], alpha = 0.3, color = "red", label = None)
                plot[2].set_title("Val Set Prediction")
                plot[2].set_ylabel("y")
                plot[2].set_xlabel(r'$t$')

            plt.sca(plot[2])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            pl.legend(**legend_font_dict)
            pl.tight_layout()

            display.display(pl.gcf())

            #clear the plot outputt and then re-plot

    
    def optimize(self, n_trust_regions = 1, max_evals = None, y = None, X=None, store_path=None, 
                       scoring_method = "mse", criterion = MSELoss(),
                       epochs = 25, learning_rate = 0.005,  
                       reparam_f = None, #ODE_criterion = None, 
                       init_conditions = None, scale = True, 
                       force = None, backprop_f = None,
                        ode_coefs = None, solve = True, tr_score_prop = 0.5, q = None, eq_system = False, 
                        nonlinear_ode = False, reg_type = "nl_ham", solve_sample_prop = 1, **kwargs): #, beta = None
        """Performs optimization (with cross-validation).

        Uses Bayesian Optimization with Gaussian Process priors to optimize ESN hyperparameters.

        .. warning::
            the :attr:`epochs`, :attr:`backprop_f`, and :attr:`learning_rate` arguments only currently only works with unsupervised training.
            This is the part of RcTorch which fits ODEs, and should only be used for non-linear equations.

        .. admonition:: And, by the way...

            :attr:`ODE_criterion`, :attr:`init_conditions`, :attr:`force`, :attr:`reg_type`, :attr:`eq_system`,
            :attr:`q`,  :attr:`nonlinear_ode`, :attr:`reparam_f` are also unsupervised arguments which should only be used
            for solving (unsupervised) differential equations.


        Parameters
        ----------
        y : numpy array
            Column vector with target values (y-values)

        x : numpy array or None
            Optional array with input values (x-values)

        store_path : str or None
            Optional path where to store best found parameters to disk (in JSON)
        
        max_evals : int
            the maximum number of RcNetworks to train

        epochs : int
             backprop training epochs

        tr_score_prop: float
            if the network is running unsupervised, this argument will allow the network to score the training set as well.
            for unsupervised (data based runs) this parameter has no effect.

        n_trust_regions : int
            This argument determines the n number of BO runs to run simultaeneoulsy. 
            RcTorch uses the Turbo-1 and Turbo-m algorithms, see `this paper <https://arxiv.org/abs/1910.01739>`_ by Uber AI.
            n total BO arms are run in parallel, and each performs local bayesian optimization which is faster and more robust
            than standard global bayesian optimization. 

        q : float
            a diffeq hp



        Returns
        -------
        best_arguments : dict
            The best parameters found during optimization

        """
        #assign attributes

        #solve_sample_prop has to do with random sampling.

        acceptable_args = ['X', 'y', 
                 'acceptable_args', 
                 'backprop_f', 
                 'criterion', 
                 'epochs', 
                 'eq_system', 
                 'force',
                 'init_conditions', 
                 'learning_rate', 
                 'kwargs',
                 'max_evals',
                 'n_trust_regions',  
                 'nonlinear_ode', 
                 'ode_coefs', 
                 'ODE_criterion',
                 'q', 
                 'reg_type',
                 'reparam_f',
                 'scale', 
                 'scoring_method', 
                 'solve', 
                 'solve_sample_prop',
                 'store_path', 
                 'tr_score_prop', 
                 '__class__']


        ################################################ START AUTOMATIC ASSIGN ATTRIBUTES ###########################################
        all_args = {**locals(), **kwargs}
        del all_args['self']

        #assign the log attributes:
        for key, val in all_args.items():
            
            if key != 'self':
                if key in acceptable_args:
                    setattr(self, key, val)
                else:
                    assert False, f'invalid argument, {key}'
            else:
                assert False, f'invalid argument, {key}'

        #assign leftover args in the acceptable_args_list as None
        entered_keys = list(all_args.keys())
        for key in acceptable_args:
            if key != '__class__':
                if key not in entered_keys:
                    setattr(self, key, None)
        ########################################## END AUTOMATIC ASSIGN ATTRIBUTES ################################################

        self.nl = nonlinear_ode

        #assign attributes to self
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)

        #check for required arguments:
        assert isinstance(n_trust_regions, int), "you must enter n_trust_regions (int)"
        assert isinstance(max_evals, int), "you must enter max_evals (int)"

        #process input data
        self.best_score_yet = None
        self.dev = {"device" : self.device, "dtype" : self.dtype, "requires_grad" : False}

        if not self.ODE_order:
            
            self.y = y = _check_y(y, tensor_args = self.dev) 
            self.x = x = _check_x(X, y, tensor_args = self.dev, supervised = True)
        else:
            self.y = None
            self.x = x = _check_x(X, y, tensor_args = self.dev, supervised = False)

        if not self.ODE_order:
            self._validate_data(y, x, self.verbose)
        
        self.n_inputs = self.x.shape[1] 
        if not self.n_outputs:
            self.n_outputs = y.shape[1]


        if type(self.bounds["n_nodes"]) != int and type(self.bounds["n_nodes"]) != float:
            self.reservoir_matrices =  self.reservoir_matrices_id = None
        elif self.reservoir_weight_dist == "normal":
            self.reservoir_matrices =  self.reservoir_matrices_id = None
        else:
            self.reservoir_matrices = ReservoirBuildingBlocks(model_type = self.model_type, 
                                                              random_seed = self.random_seed,
                                                              n_nodes = self.bounds["n_nodes"],
                                                              input_weight_type = self.input_weight_type,
                                                              device = self.device,
                                                              reservoir_weight_dist  = self.reservoir_weight_dist,
                                                              n_inputs = self.n_inputs)
            self.reservoir_matrices_id = ray.put(self.reservoir_matrices)

        

        


        self.log2 = np.log(2)


        font = {'size'   : 18}
        plt.rc('font', **font)

        
        if self.ODE_order:
            self.multiple_ICs = True if len(init_conditions[0]) > 1 else False
        # if self.ODE_order:
        #     if self.n_outputs != len(init_conditions):
        #         assert False, "n_outputs must match the len of ode_coefs and init_conds"
        
        if self.n_jobs > 1 and self.ODE_order:
            if reg_type == "driven_pop":
                custom_loss = driven_pop_loss
                force = fforce
            elif reg_type == "simple_pop":
                custom_loss = driven_pop_loss
                force = no_fforce
            elif reg_type == "ham":
                custom_loss = ham_loss
            elif reg_type == "no_reg":
                custom_loss = no_reg_loss
            elif reg_type == "elastic":
                custom_loss = elastic_loss
            elif reg_type == "hennon":
                force = no_fforce
                custom_loss = hennon_hailes_loss
            elif reg_type == "multi_attractor":
                force = no_fforce
                custom_loss = multi_attractor_loss
            elif reg_type == "dual":
                force = no_fforce
                custom_loss = dual_loss
            else:
                assert False
            self.ODE_criterion = custom_loss
            if backprop_f:
                self.backprop_f = optimize_last_layer
            else:
                self.backprop_f = None

        if self.batch_size > 1:
            self.reparam_f = freparam
        
        
        # Initialize new random state
        if self.reservoir_matrices != None:
            if self.ODE_order:
                self.reservoir_matrices.n_inputs_ = x.shape[1]
            else:
                self.reservoir_matrices.n_inputs_ = self.n_inputs #max(y.shape[1], 1) if type(x) == type(None) else x.shape[1]
            self.reservoir_matrices.gen_in_weights()

        self.random_state = torch.Generator().manual_seed(self.random_seed + 2)

        init_device = self.device if not self.windowsOS else torch.device('cpu')


        ############### beta arguments currently silenced ###################
        #beta = torch.tensor(beta) if isinstance(beta, np.ndarray) else beta

        #self.beta = beta.type(self.dtype) if beta is not None else None
        ############### beta arguments currently silenced ###################

        

        try:
            self.subsequence_length = int( len(self.x) * self.subsequence_prop)
        except:
            self.subsequence_length = int( len(self.y) * self.subsequence_prop)

        # Inform user
        if self.verbose:
            print("Model initialization and exploration run...")
            
        if self.interactive:
            self.fig, self.ax = pl.subplots(1,3, figsize = (16,4))


        declaration_args = {'activation_function' : self.activation_function,
                             'output_activation' : self.output_activation,
                            #'act_f_prime' : self.act_f_prime,
                                 #'backprop' : self.backprop,
                            
                            #'n_inputs' : self.n_inputs,
                                 #'model_type' : self.model_type,
                                 #'input_weight_type' : self.input_weight_type, 
                            'solve_sample_prop' : self.solve_sample_prop,
                            'reservoir_weight_dist' : self.reservoir_weight_dist,
                            'approximate_reservoir' : self.approximate_reservoir,
                            "device" : self.device,
                            "reservoir" : self.reservoir_matrices_id,
                            "reservoir_weight_dist" : self.reservoir_weight_dist,
                            "input_weight_dist" : self.input_weight_dist,
                            "feedback_weight_dist" : self.feedback_weight_dist,
                            }
        #assert False, f"n_outputs {declaration_args['n_outputs']}"

        train_args = {"burn_in" : self.esn_burn_in, 
                       "ODE_order" : self.ODE_order,
                       #"track_in_grad" : self.track_in_grad,
                       "force" : self.force,
                       "reparam_f" : self.reparam_f,
                       "init_conditions" : self.init_conditions,
                       "ode_coefs" : self.ode_coefs,
                       "q" : self.q,
                       "eq_system" : self.eq_system,
                       "nl" : self.nl,
                       "backprop_f" : self.backprop_f,
                       "epochs" : self.epochs,
                       "SOLVE" : self.solve, 
                       'n_outputs' : self.n_outputs,
                      #"track_in_grad" : False,
                      "init_conditions" : self.init_conditions,
                      #"SCALE" : self.scale,
                      "reparam_f" : self.reparam_f,
                      "ODE_criterion" : self.ODE_criterion
                       #"multiple_ICs" : self.multiple_ICs
                       }

        backprop_args = {"backprop_f" : self.backprop_f,
                         "epochs" : self.epochs}

        test_args = {"scoring_method" : self.scoring_method,
                     "reparam": self.reparam_f,
                     "ODE_criterion" : self.ODE_criterion}
        cv_args = {
                   "tr_score_prop" : self.tr_score_prop,
                   "log_score": self.log_score}

        self.parallel_arguments = {"declaration_args": declaration_args, #"RC" : RC,
                                   "train_args": train_args,
                                   "test_args" : test_args,
                                   "backprop_args" : backprop_args,
                                   "device" : self.device,
                                   "cv_args" : cv_args
                              }

        # if self.n_trust_regions == 1:
        #     self._errorz, self._errorz_step, self._length_progress = {0 : []}, {0 : []}, {0 : []}
        #     self._errorz["all"] = []
        #     best_hyper_parameters = self._turbo_1()
        # else:
        self._errorz, self._errorz_step, self._length_progress = {}, {}, {}
        self._errorz["all"] = []
        for i in range(self.n_trust_regions):
            self._errorz[i], self._errorz_step[i], self._length_progress[i] = [], [], []


        try:

        
            best_hyper_parameters = self._turbo_m()
        except:
            print("warning, matrix found that was not positive definite, returning best hyper-parameters found to this point")
            best_hyper_parameters = self.recover_hps()

        
        return best_hyper_parameters #X_turbo, Y_turbo, state, best_vals, denormed_ #best_arguments

    def _turbo_1(self):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        self._restart_turbo_m()
        self.n_evals = 0

        dim = len(self.free_parameters)

        self.X_turbo = torch.zeros((0, dim), device = self.device)
        self.Y_turbo = torch.zeros((0, 1), device = self.device)
        
        X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)

        self.states = {}
        self.state = self.states[0] = self._turbo_initial_samples(X_init = X_init, dim = dim, turbo_state_id = 0)        
        
        n_init = self.initial_samples
        
        #self.count = 0
        # Run until TuRBO converges
        count = 0
        while not self.state.restart_triggered: 
            
            count += 1
            print(f'count: {count}')

            self._get_cv_samples()

            # Fit a GP model
            train_Y = (self.Y_turbo - self.Y_turbo.mean()) / self.Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(self.X_turbo, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Create a batch
            X_next = generate_batch(
                state=self.state,
                model=model,
                X=self.X_turbo,
                Y=train_Y,
                batch_size=self.batch_size,
                n_candidates=min(5000, max(2000, 200 * dim)),
                num_restarts=10,
                raw_samples=512,
                acqf="ts",
                device = self.device
            )

            objective_input = (self._convert_params(X_next), 0)
            Y_next, updates_dict = self.eval_objective(objective_input) 
            self._updates( scores = Y_next, batch_dict = updates_dict)

            self.n_evals += self.batch_size

            # Update state 
            self.state = update_state(state=self.state, Y_next=Y_next)

            # Append data
            self.X_turbo = torch.cat((self.X_turbo, X_next), dim=0)
            self.Y_turbo = torch.cat((self.Y_turbo, Y_next), dim=0)
            
            # Print current status
            # print( 
            #     f"{len(self.X_turbo)}) Best score: {max(Y_next).item():.4f},  TR length: {self.state.length:.2e}" + 
            #     f" length {self.state.length}"# Best value:.item() {state.best_value:.2e},
            # )
            
            # print( 
            #     f"TR length: {self.state.length:.2e}," +  f" min length {self.state.length_min:.2e}"
            #     # + Best value:.item() {state.best_value:.2e},
            # )

            

            assert len(self._errorz[0]) == len(self._errorz_step[0]), "err len: {}, err step: {}".format(len(self._errorz[0]), len(self._errorz_step[0]) )
        else:
            display.clear_output()

        #update_state       
        #display.clear_output(wait=True) 
        #display.display(pl.gcf())
                    
        # Save to disk if desired
        if not self.store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        best_vals = self.X_turbo[torch.argmax(self.Y_turbo)]
        
        denormed_ = self._denormalize_bounds(best_vals)

        #####Bad temporary code to change it back into a dictionary
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        #log_vars = ['connectivity', 'llambda', 'llambda2', 'noise', 'regularization', 'dt']
        for var in self.log_vars:
            if var in best_hyper_parameters:
                best_hyper_parameters[var] = 10. ** best_hyper_parameters[var] 

        # Return best parameters
        return best_hyper_parameters

    def _turbo_split_initial_samples(self, X_inits, n_jobs, turbo_id_override = None):

        """This function splits and prepares the initial samples in order to get initialization done."""
        batch_size = n_jobs
        nrow = X_inits[0].shape[0]
        n_clean_batches = nrow // batch_size
        final_batch_size = nrow-n_clean_batches*n_jobs

        initial_batches = []
        turbo_iter = []

        turbo_iter += [batch_size] * n_clean_batches

        if final_batch_size != 0:
            turbo_iter += [final_batch_size]

        for turbo_id, X_init in enumerate(X_inits):

            #if there is just one that we want to update, ie we are doing a restart:
            if turbo_id_override:
                turbo_id = turbo_id_override
            
            for i in range(n_clean_batches):
                
                if len(X_init) > batch_size:
                        X_batch = X_init[ (i*batch_size) : ((i+1)*batch_size), : ]
                        initial_batches.append((self._convert_params(X_batch), X_batch, [turbo_id] * len(X_batch)))
                else:
                    if final_batch_size == 0:
                        pass
                    else:

                        X_batch = X_init[ (nrow - final_batch_size) :, : ]
                        initial_batches.append((self._convert_params(X_batch), X_batch, [turbo_id] * len(X_batch)))

        return initial_batches, turbo_iter


    def _execute_initial_parallel_batch(self, turbo_state_id):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        
        #get the dimensions of the free parameters
        dim = len(self.free_parameters)

        #initlalize the turbo state for this trust region
        self.states[turbo_state_id] = state = TurboState(dim, 
                                                         length_min = self.length_min, 
                                                         batch_size=self.batch_size, 
                                                         success_tolerance = self.success_tolerance)

        #get the initial randomly sampled points
        X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)

        #get the training and validation sets
        self._get_cv_samples()


        objective_inputs, turbo_iter = self._turbo_split_initial_samples([X_init], self.n_jobs, turbo_id_override = turbo_state_id)

        
        results = []
        for i, objective_input in enumerate(objective_inputs):
            print(i)
            result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
            #ray.wait()
            results.append(result_i)

        self.n_evals += self.initial_samples

        X_nexts, Y_nexts, batch_dicts = zip(*results)
        X_nexts, Y_nexts, batch_dicts = list(X_nexts), list(Y_nexts), list(batch_dicts)

        [self._updates(scores=result[1], batch_dict = result[2]) for i, result in enumerate(results)]

        self._update_idx_parallel(results)

        for i, X_next in enumerate(X_nexts):
            Y_next = Y_nexts[i]
            self._update_turbo(X_next = X_next, Y_next = Y_next)

    def _execute_initial_parallel_batches(self):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        
        dim = len(self.free_parameters)
        for turbo_state_id in range(self.n_trust_regions):
            self.states[turbo_state_id] = state = TurboState(dim, 
                                                             length_min = self.length_min, 
                                                             batch_size=self.batch_size, 
                                                             success_tolerance = self.success_tolerance)

        X_inits = [get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype) for i in range(self.n_trust_regions)]

        self._get_cv_samples()

        objective_inputs, turbo_iter = self._turbo_split_initial_samples(X_inits, self.n_jobs)

        results = []
        for objective_input in objective_inputs:
            result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
            results.append(result_i)

        self.n_evals += self.initial_samples * self.n_trust_regions

        X_nexts, Y_nexts, batch_dicts = zip(*results)
        X_nexts, Y_nexts, batch_dicts = list(X_nexts), list(Y_nexts), list(batch_dicts)

        [self._updates(scores=result[1], batch_dict = result[2]) for i, result in enumerate(results)]

        #self._update_idx_parallel(results)


        ids = [batch_dict["trust_region_ids"] for batch_dict in batch_dicts]
        idxs = []
        for id_set in ids:
            idxs += id_set
        idxs = torch.tensor(idxs, dtype=torch.int32, device = self.device).reshape(-1, 1)

        self._idx = torch.vstack((self._idx, idxs))

        for i, X_next in enumerate(X_nexts):
            Y_next = Y_nexts[i]
            self._update_turbo(X_next = X_next, Y_next = Y_next)

    def _updates(self, scores, batch_dict):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """

        if not self.best_score_yet:
            self.best_score_yet = batch_dict
        elif batch_dict["best_score"] < self.best_score_yet["best_score"]:
            self.best_score_yet = batch_dict
        else:
            pass   

        trust_region_ids = batch_dict["trust_region_ids"]
        
        for i, score in enumerate(scores):
            trust_region_id = trust_region_ids[i]
            
            state = self.states[trust_region_id]
            score__ = -float(score)
            if self.log_score:
                score__ = 10**score__
            self._errorz[trust_region_id].append(score__)
            self._errorz["all"].append(score__)
            self._length_progress[trust_region_id].append(state.length)

            self._errorz_step[trust_region_id] += [min(self._errorz[trust_region_id])]#* len(scores) #+= [min(self._errorz[trust_region_id])] 

    def _update_idx_parallel(self, results):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        #TODO: this function is retarded, rewrite.
        #assert False, results
        idxs = []

        for i, result in enumerate(results):
            num_points = result[0].shape[0] 
            idx_spec = result[2]["trust_region_ids"]#
            idxs += idx_spec
        idxs = torch.tensor(idxs, dtype=torch.int32, device = self.device).reshape(-1, 1)
        #assert False, results[0][0] 
        try:
            self._idx = torch.vstack((self._idx, idxs))
        except:
            assert False, results[0][0] 

    def _get_cv_samples(self):
        """
        TODO doctstring
        """

        cv_samples = [self._objective_sampler() for i in range(self.cv_samples)]

        fit_inputs = []
        val_inputs = []
        for i, cv_sample in enumerate(cv_samples):
            cv_sample_score = 0
            fit_inputs.append(cv_sample[0])
            val_inputs.append(cv_sample[1])

        self.parallel_arguments["fit_inputs"]= fit_inputs
        self.parallel_arguments["val_inputs"]= val_inputs

        # self.parallel_arguments["fit_inputs"] = fit
        #

        self.parallel_args_id = ray.put(self.parallel_arguments)

    def _convert_params(self, parameters):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        int
            Description of return value

        """
        return [self._construct_arguments(parameters[i, :]) for i in  range(parameters.shape[0])]

    def _combine_new_turbo_batches(self, sorted_lst, n_jobs, turbo_iter):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        sorted_lst : list
            a sorted list ...
        n_jobs : int
            The number of jobs to run
        turbo_iter : dtype
            Desc

        Returns
        -------
        a list:
            list(zip(hps, new_batches, new_turbo_ids))

        """
        prev_index =0
        new_batches = []
        new_turbo_ids = []
        hps = []
        for i, index in enumerate(turbo_iter):
            sub_list = sorted_lst[prev_index:index+prev_index]
            X_batch_lst, turbo_ids = zip(*sub_list)
            X_batch_lst = list(X_batch_lst)
            prev_index += index

            X_batch_spec = torch.vstack(X_batch_lst)
            hps_spec = self._convert_params(X_batch_spec)

            new_turbo_ids.append(turbo_ids)
            new_batches.append(X_batch_spec)
            hps.append(hps_spec)
        return list(zip(hps, new_batches, new_turbo_ids))

    def _update_turbo(self, X_next, Y_next) -> None:
        """
        Update the turbo state by concatenating the most recent BO round results to Y_next
        and the respective hps by concatenating X_next to X_turbo.

        Extended description of function.

        Parameters
        ----------
        X_next : pytorch.tensor
            The most recent batch of tested hps
        Y_next : pytorch.tensor
            The most recent objective function score

        Returns
        -------
        None

        """
        self.X_turbo = torch.cat((self.X_turbo, X_next), dim=0)
        self.Y_turbo = torch.cat((self.Y_turbo, Y_next), dim=0)


    def _turbo_m(self):
        """
        Runs the turbo_m algorithm, which is more robust than turbo_1

        Extended description of function.

        Parameters
        ----------
        None

        Returns
        -------
        best_hyper_parameters
            A dictionary with the best hyper-parameters

        """
        dim = len(self.free_parameters)

        self.n_evals = 0

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart_turbo_m()

        self.X_turbo = torch.zeros((0, dim), device = self.device)
        self.Y_turbo = torch.zeros((0, 1), device = self.device)

        #set up dict of turbo states
        self.states = {}

        self._execute_initial_parallel_batches()

        
        n_init = self.initial_samples

        # Run until TuRBO converges

        self.RCs_per_turbo_batch = self.n_trust_regions * self.turbo_batch_size
        self.n_normal_rounds = self.RCs_per_turbo_batch // self.n_jobs
        self.job_rounds_per_turbo_batch = self.n_normal_rounds + 1
        self.last_job_round_num_RCs = self.RCs_per_turbo_batch % self.n_jobs
        self.turbo_iter = [self.n_jobs] * self.n_normal_rounds

        if self.last_job_round_num_RCs != 0:
            
            self.turbo_iter += [self.last_job_round_num_RCs]
        
        count = 0

        while self.n_evals < self.max_evals: #not self.state.restart_triggered: 
            count += 1

            # Generate candidates from each TR
            #X_cand = torch.zeros((self.n_trust_regions, self.dim), device = self.device)
            #y_cand = torch.inf * torch.ones((self.n_trust_regions, self.n_cand, self.batch_size), device = self.device) 
            X_nexts = []
            for turbo_id, round_batch_size in enumerate(range(self.n_trust_regions)):

                idx = np.where(self._idx == turbo_id)[0] 

                sub_turbo_X = self.X_turbo[idx]
                sub_turbo_Y = self.Y_turbo[idx]


                #ensure that turbo-m is working correctly
                if turbo_id !=0:
                    assert not torch.equal(sub_turbo_X, self.X_turbo[0])

                # Fit a GP model
                train_Y = (sub_turbo_Y - sub_turbo_Y.mean()) / sub_turbo_Y.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(sub_turbo_X, train_Y, likelihood=likelihood)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)


                # print(f"sub_turbo_X {sub_turbo_X}")
                # print(f"sub_turbo_Y {sub_turbo_Y}")

                # Create a batch
                X_next = generate_batch(
                    state=self.states[turbo_id],
                    model=model,
                    X=sub_turbo_X,
                    Y=train_Y,
                    batch_size = self.turbo_batch_size,
                    n_candidates=min(5000, max(2000, 200 * dim)),
                    num_restarts=10,
                    raw_samples=512,
                    acqf="ts",
                    device = self.device
                )
                tuple_ = X_next, turbo_id, 
                X_next_lst = X_next.split(X_next.shape[1], dim = 1)
                X_next_tuple_lst = [ (i, X_next_i, turbo_id) for i, X_next_i in enumerate(X_next_lst)]
                X_nexts += X_next_tuple_lst

            X_nexts = sorted(X_nexts, key = lambda x: x[0])
            X_nexts = [ (x[1], x[2]) for x in X_nexts]

            start = time.time()
            self._get_cv_samples()
            self.parallel_trust_regions = True
            if self.parallel_trust_regions:

                objective_inputs = self._combine_new_turbo_batches(X_nexts, self.n_jobs, self.turbo_iter)

                results = []
                for objective_input in objective_inputs:
                    result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
                    results.append(result_i)

                X_nexts_mod, Y_nexts, updates_dicts  = zip(*results)
                X_nexts_mod, Y_nexts, updates_dicts  = list(X_nexts_mod), list(Y_nexts), updates_dicts

                trust_regions_ids_lst  = [dictt["trust_region_ids"] for dictt in updates_dicts]

                #objective_inputs = [(self._convert_params(batch[0]), batch[0], batch[1]) for i, batch in enumerate(X_init_processed_batches)]

                [self._updates(result[1], result[2]) for i, result in enumerate(results)]

                #self._update_idx_parallel(results)

                if self.interactive:
                    self._train_plot_update(pred_ = updates_dicts[0]["pred"], validate_y = updates_dicts[0]["y"], steps_displayed = updates_dicts[0]["pred"].shape[0])

            X_nexts_stacked = torch.vstack(X_nexts_mod)
            Y_nexts_stacked = torch.vstack(Y_nexts)

            trust_regions_ids = list(itertools.chain.from_iterable(trust_regions_ids_lst))

            lst_to_sort = [ (i, tr_id) for i, tr_id in enumerate(trust_regions_ids)]

            mask, tr_ids = zip(*sorted(lst_to_sort, key = lambda x: x[1]))
            mask = np.array(mask)

            X_nexts_batch = X_nexts_stacked[mask,:]
            Y_nexts_batch = Y_nexts_stacked[mask,:]

            for i in range(self.n_trust_regions):

                Y_next_spec = Y_nexts_batch[mask == i, :]
                X_next_spec = X_nexts_batch[mask == i, :]

                self.states[i] = update_state(state=self.states[i], Y_next=Y_next_spec)

                # Append data
                self._update_turbo(X_next = X_next_spec, Y_next = Y_next_spec)

                self._idx = torch.vstack((self._idx, torch.ones_like(Y_next_spec) * i))

                assert len(self._idx) == len(self.Y_turbo)

            self.n_evals += self.turbo_batch_size * self.n_trust_regions

            #check if states need to be restarted
            for i, state in self.states.items():
                if state.restart_triggered:

                    idx_i = self._idx[:, 0] == i

                    #remove points from trust region
                    self._idx[idx_i, 0] = -1

                    self._errorz[i], self._errorz_step[i], self._length_progress[i] = [], [], []
                    print(f"{self.n_evals}) TR-{i} is restarting from: : ... #TODO")

                    self._execute_initial_parallel_batch(i)

                    #X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)
                    
                    assert self.states[i].restart_triggered == False
                    
                    #{fbest:.4}")

                    #self._errorz_step[i] += [min(self._errorz[i])] * self.batch_size


            # # Print current status
            # print( 
            #     f"{len(self.X_turbo)}) Best score: {max(Y_next).item():.4f},  TR length: {self.state.length:.2e}" + 
            #     f" length {self.state.length}"# Best value:.item() {state.best_value:.2e},
            # )
            
            # print( 
            #     f"TR length: {self.state.length:.2e}," +  f" min length {self.state.length_min:.2e}"
            #     # + Best value:.item() {state.best_value:.2e},
            # )
        else:
            display.clear_output()

        
        #display.clear_output(wait=True) 
        #display.display(pl.gcf())
                    
        # Save to disk if desired
        if not self.store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        best_vals = self.X_turbo[torch.argmax(self.Y_turbo)]
        
        denormed_ = self._denormalize_bounds(best_vals)
        
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        best_hyper_parameters = self.convert_log_params(best_hyper_parameters)

        # Return best parameters
        return best_hyper_parameters
        

    def _restart_turbo_m(self):
        """
        TODO doctstring
        """
        self._idx = torch.zeros((0, 1), dtype=torch.int32)  # Track what trust region proposed what using an index vector
        # self.failcount = torch.zeros(self.n_trust_regions, dtype=torch.int32)
        # self.succcount = torch.zeros(self.n_trust_regions, dtype=torch.int32)
        # self.length = self.length_init * torch.ones(self.n_trust_regions)

    def convert_log_params(self, hps):
        #assert False

        vars2del = []
        vars2add = {}
        for var in hps:
            if "log_" in var:
                vars2add[var[4:]] = 10. ** hps[var] 
                vars2del.append(var)
        for var in vars2del:
            del hps[var]

        return {**hps, **vars2add}


    def recover_hps(self, alternative_index = None):
        """
        Recover best hyper-parameters from RcBayesOpt object.
        
        This is useful if your run crashed, or you put a large number of iterations and 
        are training the object in a jupyter notebook and you want to stop the run.

        This method will then recover the best hyper-parameters by extracting them from the
        :attr:`X_turbo` (list of hp values) and :attr:`y_turbo` (the respective scores), along with
        converting those HPs back to their original scale.

        Parameters
        ----------
        alternative__index: int
            the alternative_index will give you the i/ :sup:th best hyper-parameters (as opposed to the HPs with the highest score)
            this method allows you to extract them. 

        Returns
        -------
        best_hyper_parameters: dict
            a dictionary with the optimized hyper-parameters

        """
        if alternative_index:
            _, best_indices = self.Y_turbo.view(-1,).topk(len(self.Y_turbo))

            best_vals = self.X_turbo[best_indices[alternative_index]]
        else:
            best_vals = self.X_turbo[torch.argmax(self.Y_turbo)]
            
        denormed_ = self._denormalize_bounds(best_vals)

        #####Bad temporary code to change it back into a dictionaryf
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        best_hyper_parameters = self.convert_log_params(best_hyper_parameters)

        # Return best parameters
        return best_hyper_parameters