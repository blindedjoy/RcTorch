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

import types
import functools

from .defs import *

import logging
#logger = multiprocessing.log_to_stderr()
#logger.setLevel(multiprocessing.SUBDEBUG)



#https://stackoverflow.com/questions/9336646/python-decorator-with-multiprocessing-fails
class my_decorator(object):
    def __init__(self, target):
        self.target = target
        try:
            functools.update_wrapper(self, target)
        except:
            pass

    def __call__(self, candidates, args):
        f = []
        for candidate in candidates:
            f.append(self.target([candidate], args)[0])
        return f

# ideas from yesterday (trip related)
# 1. moving average RC
# 2. Penalize the time the algorithm takes to run

@dataclass
class TurboState:
    """
    This is from BOTorch. The Turbo state is a stopping condition.

    #TODO finish description and read Turbo paper

    Arguments:
        dim: 
        batch_size:
        length_min:
        length_max:
        failure_counter:
        success_counter:
        success_tolerance:
        best_value:        the best value we have seen so far
        restart_triggered: has a restart been triggered? If yes BO will terminate.
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
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def get_initial_points(dim, n_pts, device, dtype):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def update_state(state, Y_next):
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
    """generate a batch for the Bayesian Optimization

    Arguments:
        state: the TURBO state (a stopping metric)
        model: the GP (Gaussian Process) BOTorch model
        X: points evaluated (a vector of hyper-parameter values fed to the objective function) # Evaluated points on the domain [0, 1]^d in original example, not ours.
        Y: Function values
        n_candidates: Number of candidates for Thompson sampling
        num_restarts:
        raw_samples:
        acqf: acquisition function (thompson sampling is preferred)
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
    """A set of preloaded reservoir weights matching the reservoir seed and approximate sparcity (if applicable)

    Parameters
    ----------
    book: dict
        the set of approximate reservoir tensors and associated connectivity thresholds
    keys:
        the key to access the dictionary (typically an approximate sparcity threshold)
    """
    def __init__(self, book, keys):
        self.sparse_book = book
        self.sparse_keys_ = np.array(keys)

    def get_approx_preRes(self, connectivity_threshold):
        """ Given a connectivity threshold, the method will return the sparse matrix most closely matching that threshold.

        Parameters
        ----------
        connectivity_threshold: float
            #TODO description

        """
        #print("sparse_keys", self.sparse_keys_, "connectivity_threshold", connectivity_threshold   )
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  self.sparse_book[key_].clone()
        return val

class GlobalSparseLibrary:
    """
    This will approximate the search for the sparcity hyper-parameter, which will dramatically speed up training of the network.

    Parameters
    ----------
    lb: int
        lower bound (connectivity)
    ub: int
        upper bound (connectivity)
    n_nodes: number of nodes in the reservoir
    precision: the precision of the approximate sparcity metric
    flip_the_script: bool
        completely randomizes which reservoir has been selected.
    """

    def __init__(self, device, lb = -5, ub = 0, n_nodes = 1000, precision = None, 
                 flip_the_script = False):
        self.lb = lb
        self.ub = ub
        self.n_nodes_ = n_nodes
        self.library = {}
        self.book_indices = []
        self.precision = precision
        self.flip_the_script = flip_the_script
        self.device = device
        

    def addBook(self, random_seed):
        """
        Add a sparse reservoir set by looping through different different connectivity values and assigining one reservoir weight matrix per connetivity level
        and storing these for downstream use by EchoStateNetwork
        We generate the reservoir weights and store them in the sparse library.

        Parameters
        ----------
        random_seed: the random seed of the SparseLibrary with which to make the preloaded reservoir matrices
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
        """returns book indices"""
        return self.book_indices

    def get_approx_preRes(self, connectivity_threshold, index = 0):
        """ This function is for use by EchoStateNetwork to access different sets of reservoir matrices.
        Given a connectivity threshold we access a reservoir by approximate sparcity / connectivity.
        But which randomly generated reservoir we select is determined by the index, which is what ESN uses if the one reservoir is nilpotent.
        Parameters
        ----------
            connectivity threshold: float
            index: int
                which preloaded reservoir do we want to load? Each index references a difference Sparse Booklet (ie a different reservoir)

        Returns
        -------
        sparse booklet for reading downstream by EchoStateNetwork class
        (we are returning a set of pre-loaded matrices to speed up optimization of the echo-state network by avoiding 
        repeated tensor generation.)
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


    Parameters
    ----------
        model_type: either random, cyclic or delay line
        input_weight_type: exponential or uniform
        random_seed: the random seed to set the reservoir
        n_nodes: the nodes of the network
        n_inputs: the number of observers in the case of a block experiment, the size of the output in the case of a pure prediction where teacher forcing is used.
    

    """
    def __init__(self, model_type, input_weight_type, random_seed , n_nodes, n_inputs = None, Distance_matrix = None, sparse = False, device = None):
        #print("INITIALING RESERVOIR")

        #initialize attributes
        self.device = device
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
        """Generates the matrices required for generating reservoir weights"""
        gen = Generator(device = self.device).manual_seed(self.random_seed)
        n = self.n_nodes_
        self.accept = rand(n, n, **self.tensorArgs, generator = gen)
        self.reservoir_pre_weights = rand(n, n, **self.tensorArgs, generator = gen) * 2 - 1

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
        
        This method assigns the reservoir input weights for later use downstream by the EchoStateNetworkClass.

        Parameters
        ----------
        inputs : None
        Returns: None
        ----------
        
        """

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

def process_score(score__, upper_error_limit = 1000000, device = None):
                    if torch.isnan(score__):
                        score__ = tensor(upper_error_limit, device = device, requires_grad  = False, dtype = torch.float32)
                    else:
                        score__ = min(score__, tensor(upper_error_limit, device = device, requires_grad = False, dtype = torch.float32))
                    return score__

def combine_score(tr_score, val_score, tr_score_prop, log_score):
        tr_score = tr_score.type(torch.float32)
        val_score = val_score.type(torch.float32)
        if log_score:
            tr_score = torch.log(tr_score)
            val_score = torch.log(val_score)
            return tr_score * tr_score_prop + val_score * (1- tr_score_prop)
        else:
            return torch.log(tr_score * tr_score_prop + val_score * (1- tr_score_prop))

def execute_objective(arguments):
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
    #later call define_tr_val from within this function for speedup.
    #score, pred_ = self.define_tr_val() #just get the train and test...
    

    cv_samples, parallel_arguments, parameters, windowsOS, id_ = arguments
    device = parallel_arguments["device"]
    declaration_args = parallel_arguments["declaration_args"]
    backprop_args = parallel_arguments["backprop_args"]
    #del declaration_args['reservoir']

    cv_args = parallel_arguments["cv_args"]
    log_score, rounds, tr_score_prop = cv_args["log_score"], cv_args["rounds"], cv_args["tr_score_prop"]

    # if windowsOS == True:
    #     #move specific arguments to the gpu.
    #     cv_samples_ = []
    #     for i, cv_sample in enumerate(cv_samples):
    #         if  cv_sample["tr_y"].device != device:
    #             #if cv_sample["tr_x"]:
    #             train_x, validate_x = cv_sample["tr_x"].to(device), cv_sample["val_x"].to(device)

    #             train_y, validate_y  = cv_sample["tr_y"].to(device), cv_sample["val_y"].to(device)
    #         else:
    #             #consider not sending the cv_sample["x"] if it's a pure prediction.
    #             train_x, train_y = cv_sample["tr_x"], cv_sample["tr_y"]
    #             validate_x, validate_y = cv_sample["val_x"], cv_sample["val_y"]
    #         cv_samples_.append((train_x, train_y, validate_x, validate_y))

    #         #now move the input weights and the reservoir arguments to the gpu.
    #         #deepcopy()
    #         assert declaration_args["reservoir"] != None
    #         if declaration_args["reservoir"] != None:
    #             declaration_args["reservoir"].in_weights = declaration_args["reservoir"].in_weights.to(device)
    #             declaration_args["reservoir"].accept = declaration_args["reservoir"].accept.to(device)
    #             declaration_args["reservoir"].reservoir_pre_weights = declaration_args["reservoir"].reservoir_pre_weights.to(device)

    #     RC = EchoStateNetwork(**declaration_args,  **parameters, id_ = id_)
    
    RC = EchoStateNetwork(**declaration_args, **parameters, id_ = id_)
    cv_samples_ = []
    for i, cv_sample in enumerate(cv_samples):
        cv_samples_.append((cv_sample["tr_x"],  cv_sample["tr_y"], cv_sample["val_x"], cv_sample["val_y"]))

    ODE_order = parallel_arguments["train_args"]["ODE_order"]

    train_args = parallel_arguments["train_args"]
    test_args = parallel_arguments["test_args"]

    total_score = 0

    backprop_f = backprop_args["backprop_f"]

    if type(train_args["init_conditions"][0]) == list:
        multiple_ICs = True if len(train_args["init_conditions"][0]) > 1 else False
    else:
        multiple_ICs = False

    for i, cv_sample in enumerate(cv_samples_):
        #print(i)
        train_x, train_y, validate_x, validate_y = cv_sample

        cv_sample_score = 0

        #divis = rounds *2
        if ODE_order:
            if multiple_ICs:
                # if train_score:
                #         return {"scores" : scores, 
                #                 "weights": gd_weights, 
                #                 "biases" : gd_biases,
                #                 "ys"     : ys,
                #                 "ydots"  : ydots,
                #                 "losses" : Ls}
                dictt = RC.fit(X = train_x, y = train_y, train_score = True, **train_args)
                train_scores = dictt["scores"]
                val_scores, pred_, id_ = RC.test(X=validate_x, y= validate_y, **test_args)

                
                for i, train_score in enumerate(train_scores):
                    
                    train_scores[i] = process_score(train_score, device = device)

                    val_score = process_score(val_scores[i], device = device)# / divis

                    round_score = combine_score(train_score, val_score, tr_score_prop, log_score)  
                    cv_sample_score += round_score 
                total_score += cv_sample_score

            else:
                #train_score = RC.fit(X = train_x, y = train_y, train_score = True, **parallel_arguments["train_args"])
                results = RC.fit(X = train_x, y = train_y, train_score = True, **train_args)
                train_scores = results["scores"]
                train_score = train_scores[0]

                train_score = process_score(train_score, device = device)# / divis

                val_scores, pred_, id_ = RC.test(X=validate_x, y= validate_y, **test_args)
                val_score = process_score(val_scores[0], device = device)# / divis

                cv_sample_score = combine_score(train_score, val_score, tr_score_prop, log_score)  

            
        else:
            _ = RC.fit(X = train_x, y = train_y,**parallel_arguments["train_args"])
            #train_score = process_score(train_score)
            
            val_score, pred_, id_ = RC.test(X=validate_x, y= validate_y, **parallel_arguments["test_args"])
            val_score = process_score(val_score)

            cv_sample_score = val_score #combine_score(train_score, val_score)

        #del train_x; del train_y;
        #del validate_x;

        if id_ != 0:
            del validate_y; del pred_;

        total_score += cv_sample_score
        #if device == torch.device('cuda'):
        #    torch.cuda.empty_cache()

    #del RC;
    #score_mu = total_score/len(cv_samples)
    #del cv_samples;
    nn = len(cv_samples_) * len(train_scores)

    total_score = total_score / nn
    if id_ == 0:
        try:
            validate_y = validate_y.to("cpu")
        except:
            pass
        return float(total_score), {"pred": pred_.to("cpu"), "val_y" : validate_y}, id_
    else:
        return float(total_score), None, id_

def sech2(z):
    return (1/(torch.cosh(z)))**2

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

    device : string or torch device                                 #TODO flexible implimentation
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
    model_type: string
            #TODO
        right now it is unclear whether this means reservoir type or model type.
        likely that is unclear because I haven't implimented cyclic or exponential here. #TODO impliment uniform and expo weights
    BOTorch Turbo bayesian optimization parameters
    ----------------------------------------------
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

    #### uncompleted tasks:
    #1) upgrade to multiple acquisition functions.
    #self.acquisition_type = acquisition_type
    ######
    """
    def __init__(self, bounds, subsequence_length, model=EchoStateNetwork, initial_samples=50,
                 validate_fraction=0.5, steps_ahead=None, batch_size=1, cv_samples=1,
                 scoring_method='nrmse', esn_burn_in=0, random_seed=None, esn_feedback=None, 
                 verbose=True, model_type = "random", activation_function = nn.Tanh(), 
                 input_weight_type = "uniform", interactive = False, 
                 approximate_reservoir = False, length_min = 2**(-9), 
                 device = None, success_tolerance = 7, dtype = torch.float32,
                 windowsOS = False, track_in_grad = False, patience = 400, ODE_order = None,
                 dt = None, log_score =  False, act_f_prime = sech2
                 ):
        self.dt = dt
        self.log_score = log_score

        self.ODE_order = ODE_order
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

        self.track_in_grad = track_in_grad
        self.patience = patience

        print("FEEDBACK:", esn_feedback, ", device:", device)

        #turbo variables
        self.batch_size = batch_size
        self.length_min = length_min
        self.success_tolerance = success_tolerance

        #interactive plotting for jupyter notebooks.
        self.interactive = interactive
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
        self.esn_burn_in =  tensor(esn_burn_in, dtype=torch.int32).item()   #torch.adjustment required
        self.esn_feedback = esn_feedback

        self.seed = random_seed
        self.feedback = esn_feedback
        self.verbose = verbose
        self.model_type = model_type

        #self.Distance_matrix = Distance_matrix

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.parameters)
        
        self.activation_function = activation_function
        self.act_f_prime = act_f_prime

        self.input_weight_type = input_weight_type
        
        if "n_nodes" in self.bounds:
            if type(self.bounds["n_nodes"]) != int and type(self.bounds["n_nodes"]) != float:
                self.reservoir_matrices = None
                assert False, type(self.bounds["n_nodes"])
            else:
                self.reservoir_matrices = ReservoirBuildingBlocks(model_type = self.model_type, 
                                                              random_seed = self.seed,
                                                              n_nodes = self.bounds["n_nodes"],
                                                              input_weight_type = self.input_weight_type,
                                                              device = self.device)
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
        return scaled_bounds, tensor(scalings, device = self.device, requires_grad = False), tensor(intercepts, device = self.device, requires_grad = False) #torch.adjustment required

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
        

        self.log_vars = ['connectivity', 'llambda', 'llambda2', 'enet_strength',
                         'noise', 'regularization', 'dt', 'gamma_cyclic' 
                         ]

        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value
            # if name in self.log_vars:
            #     arguments[name] = 10. ** value
            # else:
                

        for var in self.log_vars:
            if var in arguments:
                arguments[var] = 10. ** arguments[var]  # Log scale correction

        #assert False, f'args {arguments}'

        if 'n_nodes' in arguments:
            arguments['n_nodes'] = tensor(arguments['n_nodes'], dtype = torch.int32, device = self.device, requires_grad = False)  # Discretize #torch.adjustment required

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
    
    # def eval_objective(self, x):
    #     """This is a helper function we use to unnormalize and evaluate a point"""
    #     return self.HRC(x) 
    #     #original BoTorch code:
    #     #unnormalize(x, self.scaled_bounds))

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
        esn.fit(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test2(x=validate_x, y=validate_y, scoring_method=self.scoring_method, 
                            steps_ahead=self.steps_ahead, alpha=self.alpha)

        return score

    def define_tr_val(self, inputs):
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
        ####
        #print("Hayden edit: parameters: " + str(parameters))
        #print("Hayden edit: fixed parameters: " + str(self.fixed_parameters))
        #print("Hayden edit: free parameters: " + str(self.free_parameters))
        ####
        
        start_index, random_seed = inputs["start_index"], inputs["random_seed"]
        train_stop_index = start_index + self.train_length
        validate_stop_index = train_stop_index + self.validate_length
        
        # Get samples
        if self.ODE_order:
            train_y = None
            validate_y = None
        else:
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
        """This function builds a list of dictionairies with unique keys.

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

    def objective_sampler(self):
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

        
        # Set viable sample range
        if not self.ODE_order:
            #assert False
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
        self.validate_length = torch.round(tensor(self.subsequence_length * self.validate_fraction, requires_grad  = False)).type(torch.int32)
        self.train_length = self.subsequence_length - self.validate_length

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
                         cat([valid[i], tensor(pred_[i,:])],0), color = "red", alpha = 0.3)
        #ax.set_xlim(start_loc, 2200)

        plt.legend()
    
    def train_plot_update(self, pred_, validate_y, steps_displayed, elastic_losses = None, restart_triggered = False):
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
            #log2 = np.log(2)

            # Plot 1: the training history of the bayesian optimization
            # if not self.log_score:
            #     self.ax[0].plot(np.log(self.errorz_step), alpha = 0.5, color = "blue", label = labels[0] )
            #     self.ax[0].plot(np.log(self.errorz), alpha = 0.2, color = "green", label = labels[1])
            # else:
            font_dict = {"prop" : {"size":14}}
            ticks_font_size = 14

            self.ax[0].plot(10**np.array(self.errorz_step), alpha = 0.5, color = "blue", label = labels[0] )
            self.ax[0].plot(10**np.array(self.errorz), alpha = 0.2, color = "green", label = labels[1])
            #self.ax[0].set_title("log(error) vs Thompson Sampling step")
            self.ax[0].set_ylabel(f"log({self.scoring_method})")
            self.ax[0].set_xlabel("BO step")
            self.ax[0].set_ylabel("Error")
            self.ax[0].legend(**font_dict)
            self.ax[0].set_yscale("log")
            #self.ax[0].set_ylim(10**-8,1)
            # self.ax[0].set_xtickslabels(fontsize= ticks_font_size )
            # self.ax[0].set_ytickslabels(fontsize= ticks_font_size )
            
            #plot 2: the turbo state            #plot 2: the turbo state
            self.ax[1].axhline(np.log(self.state.length_max)/self.log2, color = 'green', label = 'max length')
            self.ax[1].set_title("TURBO state")
            self.ax[1].plot(np.log(self.length_progress)/self.log2, color = 'blue', label = 'current length')
            self.ax[1].axhline(np.log(self.state.length_min)/self.log2, color = 'red', label = 'target length')
            self.ax[1].legend()  

            #plot 3 (most recent prediction)
            self.ax[2].clear()
            if not self.ODE_order:
                self.ax[2].plot(validate_y_2plot.to("cpu"), alpha = 0.5, color = "blue", label = "val set") #[:steps_displayed]
                self.ax[2].set_ylim(self.y.min().item() - 0.1, self.y.max().item() )
            self.ax[2].plot(pred_2plot[:steps_displayed], alpha = 0.3, color = "red", label = "latest pred")
                
            self.ax[2].set_title("Val Set Prediction")
            self.ax[2].set_ylabel("y")
            self.ax[2].set_xlabel(r'$t$')

            pl.legend()
            pl.tight_layout()

            display.display(pl.gcf())

            #clear the plot outputt and then re-plot
             

    def eval_objective(self, parameters, plot_type = "error", *args):
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
        # parameter_lst = []
        # for i in range(parameters.shape[0]):
        #     parameter_lst.append(self.construct_arguments(parameters[i, :]))

        parameter_lst = [self.construct_arguments(parameters[i, :]) for i in  range(parameters.shape[0])]

        #assert False, 'param_list ' + str(parameter_lst)
        
        start = time.time()

        #the algorithm is O(n) w.r.t. cv_samples.
        cv_samples = [self.objective_sampler() for i in range(self.cv_samples)]
        
        data_args = [(cv_samples, self.parallel_arguments, params, self.windowsOS, i) for i, params in enumerate(parameter_lst)]

        num_processes = parameters.shape[0]

        if self.batch_size > 1:
            Pool = mp_Pool(num_processes)

            #get the asynch object:
            results = Pool.map(execute_objective, data_args)
            
            Pool.close()
            Pool.join()
            results = sorted(results, key=lambda x: x[2]) 
            #try:
            Pool = mp_Pool(num_processes)

            #get the asynch object:
            results = Pool.map(execute_objective, data_args)
            
            Pool.close()
            Pool.join()
            results = sorted(results, key=lambda x: x[2]) 
            # except:
            #     #assert False
            #     results = [execute_HRC((cv_samples, parallel_arguments, params, self.windowsOS, i)) for i, params in enumerate(parameter_lst)]

            results = [(result[0], result[1]) for result in results]
            scores, preds = list(zip(*results)) 
        else:
            results = execute_objective(data_args[0])
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
        if self.interactive : #and preds[0]["id_"] == 1
            self.train_plot_update(pred_ = preds[0]["pred"], validate_y = preds[0]["val_y"], 
                steps_displayed = preds[0]["pred"].shape[0]) #l2_prop  = self.l2_prop) #elastic_losses = RC.losses, 

        Scores_ = tensor(Scores_, dtype = self.dtype, device = self.device, requires_grad = False).unsqueeze(-1)

        #print('success_tolerance', self.state.success_counter)
        #print('failure_tolerance', self.state.failure_counter)
        #score_str = 'iter ' + str(self.count) +': Score ' + f'{error.type(torch.float64).mean():.4f}'# +', log(\u03BC):' + f'{log_mu:.4f}' 

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
    
    
    
    def optimize(self, y = None, x=None, store_path=None, epochs = 25, learning_rate = 0.005, scoring_method = "mse", criterion = MSELoss(), 
                    reparam_f = None, ODE_criterion = None, init_conditions = None, scale = True, force = None, backprop_f = None, backprop = False,
                     ode_coefs = None, solve = True, rounds = None, tr_score_prop = 0.5, q = None, eq_system = False, n_outputs = None, 
                     nonlinear_ode = False, reg_type = "nl_ham"):
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
        font = {'size'   : 18}
        plt.rc('font', **font)
        
        #self.multiple_ICs = True if len(init_conditions[0]) > 1 else False
        self.n_outputs = n_outputs
        if n_outputs != len(init_conditions):
            assert False, "n_outputs must match the len of ode_coefs and init_conds"
        self.nl = nonlinear_ode
        self.log2 = np.log(2)
        self.q = q
        self.eq_system = eq_system
        self.rounds = rounds
        self.tr_score_prop = tr_score_prop
        self.solve = solve
        self.ode_coefs = ode_coefs
        if self.batch_size > 1:
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
                custom_loss = hennon_hailes_loss
            else:
                assert False
            self.ODE_criterion = custom_loss
            if backprop_f:
                self.backprop_f = optimize_last_layer
            else:
                self.backprop_f = None
            self.epochs = epochs
        else:
            self.ODE_criterion = ODE_criterion
            self.backprop_f = backprop_f

        self.init_conditions = init_conditions
        
        self.scale = scale

        if self.batch_size > 1:
            self.reparam_f = freparam
            #self.force = fforce
        else:
            self.reparam_f = reparam_f
        self.force = force

        self.scoring_method = scoring_method
        self.criterion = criterion
        self.epochs = epochs
        self.learning_rate = learning_rate

        """
        if self.ODE: 
            if self.dt != None:
                #self.alpha = self.leaking_rate[0] / self.dt

                start, stop = float(x[0]), float(x[-1])
                nsteps = int((stop - start) / self.dt)
                x = torch.linspace(start, stop, steps = nsteps, requires_grad=False).view(-1,1).to(self.device)
            elif type(x) == type([]) and len(X) == 3:
                x0, xf, nsteps = x #6*np.pi, 100
                x = torch.linspace(x0, xf, steps = nsteps, requires_grad=False).view(-1,1).to(self.device)
            else:
                assert False, "Please input start, stop, dt"
        else:
            #ensure that X is a two dimensional tensor, or if X is None declare a tensor.
            X = check_x(X, y, self.dev).to(self.device)
        """
        
        # Checks
        if not self.ODE_order:
            self.validate_data(y, x, self.verbose)
        
        # Initialize new random state
        if self.reservoir_matrices != None:
            if self.ODE_order:
                
                
                self.reservoir_matrices.n_inputs_ = x.shape[1]

            else:

                self.reservoir_matrices.n_inputs_ = max(y.shape[1] - 1, 1) if type(x) == type(None) else x.shape[1]
            
            self.reservoir_matrices.gen_in_weights()

        self.random_state = Generator().manual_seed(self.seed + 2)

        init_device = self.device if not self.windowsOS else torch.device('cpu')

        if not self.ODE_order:
            if type(y) == np.ndarray:
                y = torch.tensor(y, device = init_device, requires_grad = False)
            if len(y.shape) == 1:
                y = y.view(-1, 1)
            if y.device != self.device:
                y = y.to(init_device)
            self.y = y.type(self.dtype)  
        else:
            self.y = None

        self.x = x.type(self.dtype) if x is not None else None #torch.ones(*y.shape)

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


        declaration_args = {'activation_f' : self.activation_function,
                            'act_f_prime' : self.act_f_prime,
                                 #'backprop' : self.backprop,
                            'n_outputs' : self.n_outputs,
                                 #'model_type' : self.model_type,
                                 #'input_weight_type' : self.input_weight_type, 
                            'approximate_reservoir' : self.approximate_reservoir,
                            "device" : self.device,
                            "reservoir" : self.reservoir_matrices
                            }
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
        cv_args = {"rounds" : self.rounds,
                   "tr_score_prop" : self.tr_score_prop,
                   "log_score": self.log_score}

        self.parallel_arguments = {"declaration_args": declaration_args, #"RC" : RC,
                                   "train_args": train_args,
                                   "test_args" : test_args,
                                   "backprop_args" : backprop_args,
                                   "device" : self.device,
                                   "cv_args" : cv_args
                              }

        X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)
        
        if len(X_init) > self.batch_size:
            for i in range(X_init.shape[0] // self.batch_size):
                #print(i)
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

        self.X_turbo, self.Y_turbo =  X_turbo, Y_turbo
        
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
            self.state = update_state(state=self.state, Y_next=Y_next)

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

            #assert len(self.errorz) == len(self.errorz_step), "err len: {}, err step: {}".format(len(self.errorz), len(self.errorz_step) )
        else:
            display.clear_output()

        
        #display.clear_output(wait=True) 
        #display.display(pl.gcf())
                    
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        best_vals = X_turbo[torch.argmax(Y_turbo)]

        
        
        denormed_ = self.denormalize_bounds(best_vals)
        
        try:
            denormed_ = denormalize_bounds(best_vals)
        except:
            print("FAIL")

        best_vals = X_turbo[torch.argmax(Y_turbo)]

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
        return best_hyper_parameters #X_turbo, Y_turbo, state, best_vals, denormed_ #best_arguments