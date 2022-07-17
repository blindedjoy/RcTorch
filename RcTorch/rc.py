# For more info about pytorch elastic net regularization:
#https://github.com/jayanthkoushik/torch-gel

#Imports
import math
from dataclasses import dataclass
import torch

#botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
#from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from decimal import Decimal

#gpytorch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

#torch (we import functions from modules for small speed ups in performance)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd import Function as Function
from torch.quasirandom import SobolEngine

from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.nn import Linear, MSELoss, Tanh, NLLLoss, Parameter

#other packages
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
from sklearn.linear_model import ElasticNet

#rctorchprivate code:
from .custom_loss import *
from .defs import *
#from .activations import *
from .activations import _nrmse, _convert_activation_f, _sech2, _sigmoid_derivative, _sinsq, _inverse_hyperbolic_tangent, _sech2_, _identity, _neg_sin, _neg_double_sin, _double_cos, _my_relu_i, _rnn_relu, _my_relu_i_prime, _rnn_relu_prime, _my_relu_i_prime, _rnn_relu_prime, _log_sin, _log_sin_prime, _sin2, _sin2_derivative, _sincos, _sincos_derivative 
from .rc_helpers import _check_x, _check_y, _printc, _convert_ode_coefs

#from .differentiate import _dfx
import re
import matplotlib.gridspec as gridspec
from .backprop import *

from copy import deepcopy


class RcNetwork(nn.Module):
    r"""
    Class with all functionality to train Reservoir Computers (RCs).
    Builds and trains RC networks with the specified parameters.
    In training (fitting), testing and predicting, X is a matrix consisting of column-wise time series features.
    Y is a zero-dimensional target vector or a matrix consisting of a matrix of column-wise time series vectors.
    
    The evolution of the RC is governed by the following formula:
    :math:`\bf{h}_k = \left(1-\alpha \right)\bf{h}_{k-1} + \alpha \phi \left( \bf{W}_\text{res} \cdot \bf{h}_{k-1}+\bf{W}_\text{in}\cdot \bf{u} + \bf {b}  \right)`
    where :math:`\bf{h}_k` is the k\ :sup:`th` hidden state, :math:`\alpha` is the leaking rate,
    :math:`\phi` is the activation function, :math:`\bf{W}_\text{res}` is the matrix of reservoir weights (the adjacency matrix which determines the structure of the reservoir),
    :math:`\bf{W}_\text{in}` is the set of input weights, :math:`\bf{u}` is the input and :math:`\bf{b}` is the bias.


    .. warning ::
        The :class:`RcBayes` class trains many individual :class:`RcNetwork` instances.

    .. note::
        The most important methods are the :meth:`fit`, :meth:`predict`, and :meth:`test`.
    ..
        an alernative way to use readthedocs autodoc is to put the method doctrings in the same docstring as
        as the main class. We decided to just put a docstring in each public method.
        Methods
        -------
        fit(y, x=None, burn_in=100)
            Train an Echo State Network
            test(y, x=None, y_start=None, scoring_method='mse', alpha=1.)
            Tests and scores against known output
            predict(n_steps, x=None, y_start=None)
            Predicts n values in advance
    
    To see the hidden states check out the :attr:`state` or run the :meth:`plot_states` method.

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
    backprop: bool
        if true the network initiates backpropogation.
    classification: bool
        if true the network assumes a categorical response, initiates backprop. Not yet working.
    criterion: torch.nn.Loss function
        loss function for backprogation training
    epochs: int
        the number of epochs to train the network for.
    l2_prop: float (between 0 and 1)
        this is the proportion of the l2 norm. if 1, ridge regression. if 0, lasso. in between it's elastic net regularization.
        **Please note that a significant slowdown will occur with values other than 0**
    enet_alpha: float (between 0 and 1)
        to be used in the context of solving ODEs (unsupervised). Represents the proportion of the 
        elastic net loss that is L2. Specifically the :func:`rctorchprivate.defs.elastic_loss` criterion uses the hyper-parameter in the following way:
        ``L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)``
    enet_strength : 
        to be used in the context of solving ODEs (unsupervised). Represents the strength of the
        elastic net regularization. Specifically the :func:`rctorchprivate.defs.elastic_loss` criterion uses the hyper-parameter in the following way:
        ``L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)``
    n_inputs : int
        the number of observers (input time-series) that are being input to the model.
    n_outputs : int
        the number of time series that the model will try to fit.
    input_weight_dist: str
        The probability distribution from which the input weights are drawn. 
        Valid values include `"uniform"`, and `"discrete"`
    input_weight_dist: str
        The probability distribution from which the reservoir weights are drawn.  
        Valid values include `"uniform"`, `"discrete"`, and `"normal"`.
    output_activation : str
        the output activation function. Valid values include `"identity"` (the default), `"tanh"`, and `"sin"`.
        Currently only `"identity"` and `"tanh"` are recommended. Specifically if it is known that the output is bounded,
        generally in a similar range to the training set, then `"tanh"` is recommended otherwise use the `"identity"`.
    gamma_cyclic : float
        gamma hyper-parameter to be passed to torch.optim.lr_scheduler.CyclicLR, only used in certain 
        unsupervised loss functions. Check out the defs.py file where this HP is currently in use in 
        the :func:`rctorchprivate.defs.optimize_last_layer` function.
    input_connectivity : float
        connectivity = (1- sparcity) of the input weights (number of non null weights)
    feedback_connectivity : float
        connectivity = (1 - sparcity) of the feedback weights (number of non null weights)
    noise: float
        random normal noise will be added with the following shape 
        torch.normal(0, 1, size = (self.n_nodes, t)) * self.noise
        it will be added as :math:`\epsilon` in the equation:
        :math:`\bf{h}_k = \left(1-\alpha \right)\bf{h}_{k-1} + \alpha f \left( \bf{W}_\text{res} \cdot \bf{h}_{k-1}+\bf{W}_\text{in}\cdot \bf{u} + \bf {b} + \epsilon \right)`
    """
    def __init__(self, 
                 n_nodes = 1000, 
                 bias = 0, 
                 connectivity = 0.1, 
                 leaking_rate = 0.99, 
                 spectral_radius = 0.9,                  #<-- activation, feedback
                 input_scaling = 0.5, 
                 feedback_scaling = 0.5, 
                 activation_function = "tanh", 
                 output_activation = "identity", 
                 input_weight_dist = "uniform", 
                 reservoir_weight_dist = "uniform", 
                 solve_sample_prop = 1,
                 feedback_weight_dist = "uniform", 
                 feedback = False, 
                 l2_prop = 1, 
                 random_state = 123, 
                 approximate_reservoir = False,
                 **kwargs): #<-- this line is backprop arguments #beta = None
        super().__init__()

        #The following arguments were moved from the __init__ default args for the sake of cleanly docs.
        #TODO downsize default arguments
        acceptable_args = [
                 'acceptable_args', 'approximate_reservoir', 'activation_function', 'bias', 
                 'classification',
                 'connectivity', 'device',  'dt', 'dtype', 'enet_alpha', 'enet_strength', 'feedback', 'feedback_connectivity',
                 'feedback_scaling', 'feedback_weight_dist', 'gamma', 'gamma_cyclic', 'id_', 'input_connectivity', 
                 'input_scaling', 'input_weight_dist',
                 'leaking_rate', 'l2_prop', 'mu', 'n_inputs', 'n_outputs', 'n_nodes', 'noise', 'output_activation',
                 'random_state', 'regularization', 'reservoir', 'reservoir_weight_dist','sigma', 'spectral_radius',
                 'solve_sample_prop', 'spikethreshold',  
                 'kwargs',
                 '__class__']

        ################################################ START AUTOMATIC ASSIGN ATTRIBUTES ###########################################
        all_args = {**locals(), **kwargs}
        del all_args['self']

        #assign the log attributes:
        for key, val in all_args.items():
            
            #if key != 'self':
            if key in acceptable_args:
                setattr(self, key, val)
            else:
                split_key = key.split('_')
                if split_key[0] == 'log':
                    if split_key[1] in acceptable_args:
                        #assign the attribute if it is acceptable args
                        setattr(self, split_key[1], 10**val)
                        continue
                    else:
                        assert False, f'invalid argument, {key}'
                else:
                    assert False, f'invalid argument, {key}'
                assert False, f'invalid argument, {key}'

        #assign leftover args in the acceptable_args_list as None
        entered_keys = list(all_args.keys())
        for key in acceptable_args:
            if key not in entered_keys:
                setattr(self, key, None)

        #finally assign the rest of the kwargs for good measure.
        np.random.seed(random_state)
        # for key, val in kwargs.items():
        #     if key != 'self':
        #         setattr(self, key, val)
        ########################################## END AUTOMATIC ASSIGN ATTRIBUTES ################################################


        ################################################ START MANUAL ASSIGN ATTRIBUTES ###########################################
        
        # hyper-parameters:
        torch.manual_seed(random_state)

        #set the device
        self.device = torch_device("cuda" if cuda_is_available() else "cpu")

        #a convenient dictionary to automatically assign key attributes to tensors.
        self.dev = {"device" : self.device, "dtype" : self.dtype, "requires_grad" : False}

        
        self.n_nodes = int(self.n_nodes)
        self.leaking_rate = [leaking_rate, 1 - leaking_rate]

        self.leaking_rate_orig = deepcopy(self.leaking_rate)
        #https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        #self.classification = classification

        if self.enet_alpha:
            assert self.enet_strength > 0
            assert self.enet_alpha >= 0 and self.enet_alpha <=1

        if self.reservoir_weight_dist == "uniform":
            if isinstance(self.mu, float) or isinstance(self.sigma, float):
                if self.mu != 0:
                    assert False, "to use mu and sigma hps use reservoir_weight_dist = 'normal'"

        # random state and default tensor arguments
        self.random_state = torch.Generator(device=self.device).manual_seed(random_state)
        self.no_grad_ = {"requires_grad" : False}
        self.tensor_args = {"device": self.device, "generator" : self.random_state, **self.no_grad_}

        ################################################ END MANUAL ASSIGN ATTRIBUTES ###########################################


        ################################## START ASSIGN RESERVOIR ACTIVATION FUNCTIONS #################################
        if type(activation_function) == str:

            #assign a single activation function
            self.activation_function, self.act_f_prime = _convert_activation_f(activation_function)

        elif type(activation_function) == list:
            
            #assign multiple activation functions from a list

            self._act_fs = [_convert_activation_f(act_f, derivative = False, both = False) for act_f in activation_function]
            self._act_f_primes = [_convert_activation_f(act_f, derivative = False, both = False) for act_f in activation_function]
            #mask = torch.tensor(np.random.choice(list(range(len(self._act_fs))), size = n_nodes)) #(torch.tensor(np.ones(n_nodes)) * torch.rand(250) < 0.5)*1
            n_fs = list(range(len(self._act_fs)))
            self._act_mask = torch.tensor(np.random.choice(n_fs, size = self.n_nodes))
            self.activation_function = self._multiple_act_f
            self.act_f_prime = self._multiple_act_f_prime

        elif type(activation_function) == dict:

            #assign multiple activation functions from a dict

            self.activation_function = self._multiple_act_f
            self._act_fs = []
            self._act_f_primes = []

            #probabilities
            probs = []

            for act_f,  prop in activation_function.items():

                #prop stands for proportion, we will normalize it to a probability later.

                #get the activation functions
                self._act_fs.append(_convert_activation_f(act_f, derivative = False, both = False))

                #get the derivative of the activation functions (key for unsupervised differential equation solving)
                self._act_f_primes.append(_convert_activation_f(act_f, derivative = False, both = False))
                probs.append(prop)

            probs = np.array(probs)
            probs = probs / np.sum(probs)
            n_fs = list(range(len(self._act_fs)))
            self._act_mask = torch.tensor(np.random.choice(n_fs, size = self.n_nodes, p = probs))
        else:
            assert False, f'inproper activation function input'

        ################################## END ASSIGN RESERVOIR ACTIVATION FUNCTIONS #################################

        

        ################################## START ASSIGN OUTPUT ACTIVATION FUNCTION #################################
        if output_activation == "identity":
            self.output_f, self.output_f_inv  = _identity, _identity
            self._normalize = False
        elif output_activation == "tanh":
            self._normalize = True
            self.output_f, self.output_f_inv = torch.tanh, _inverse_hyperbolic_tangent
        elif output_activation == "sin":
            self._normalize = True
            self.output_f, self.output_f_inv = torch.sin, torch.asin
        else:
            assert False, f"output_activation {self.output_f} not yet implimented"
        ################################## END ASSIGN OUTPUT ACTIVATION FUNCTION #################################

        #set up Neural network layers:

        #Reservoir layer
        self.LinRes = Linear(self.n_nodes, self.n_nodes, bias = False)  
        self.LinOut = None 
        
        with torch.no_grad():
            self._gen_reservoir()

        #TODO generalize the following: or do we need to?
        try:
            if self.regularization == None:
                self.regularization = self.log_regularization**10
        except: 
            assert False, f"You must enter either \'regularization\' or \'log_regularization as an argument to the RcNetwork.__init__ method\'"



    def __repr__(self):
        #string representation of RcNetwork
        n_nodes = str(self.n_nodes)
        connect = str(self.connectivity)
        spect   = str(self.spectral_radius)

        strr = "{" + f"n_nodes : {n_nodes}, connectivity : {connect}, spectral_radius : {spect}" + "}"
        return f"EchoStateNetwork: " + strr

    def forward(self, extended_states):
        output = self.LinOut(extended_states)
        return output

    def display_in_weights(self):
        """
        Plot a heatmap of the input weights.
        """
        sns.heatmap(self.in_weights)

    def display_out_weights(self):
        """
        Plot a heatmap of the output weights.
        """
        sns.heatmap(self.out_weights)

    def display_res_weights(self):
        """
        Plot a heatmap of the reservoir weights.
        """
        sns.heatmap(self.weights)

    def plot_states(self, n= 10):
        """
        Plots up to n hidden states.

        :: note::
            The :attr:`state` is being plotted.

        Parameters
        ----------
        n : int
            Number of hidden states to plot

        """
        for i in range(n):
            plt.plot(list(range(len(self.state[:,i]))), RC.state[:,i], alpha = 0.8)

    def fit(self, 
            y = None, 
            X = None, 
            burn_in = 0, 
            criterion =  MSELoss(), 
            
            return_states = False, 
            optimizer = None, 
            out_weights = None, 
            ODE_order = None, 
            SOLVE = True,
            reparam_f = None, 
            init_conditions = None, 
            force = None,
            ode_coefs = None, 
            train_score = False, 
            ODE_criterion = None, 
            q = None, 
            eq_system : bool = None, 
            backprop_f = None, 
            epochs = None, 
            nl : bool = False,
            n_inputs : int = None, 
            n_outputs : int = None, 
            random_sampling : bool = False, 
            sample_timepoints : int = 100,
            verbose : bool = False,
            backprop_args = None ): #beta = None, 
        
        """
        Train the network by fitting the hidden states and then solving for the output weights.

        Extended description of function.

        .. note::
            To see the hidden states check out the :attr:`extended_states` attribute!

        .. warning::
            If you enter ODE_order >=1  then the RC will perform unsupervised differential equation solving.


        Parameters
        ----------
        y : pytorch.tensor or numpy.array
            Target
        X : pytorch.tensor or numpy.array
            Observers
        burn_in : int
            number of initial steps to throw away, similar to role in Markov Chain Monte Carlo simulations
        criterion : dtype
            loss function
        epochs : int
            the number of epochs to train
        return_states : bool
            if True return the hidden states 
        nl : bool
            If True then the network tries to solve the Bernoulli differential eq family
        eq_system : bool
            If True then the network tries to solve a system of differential equations
        n_inputs : int
            the number of input timeseries, if None then the network will use teacher forcing with a pure prediction
        n_outputs : int
            the number of output timeseries
        backprop_f : function
            the backpropagation loss function
        ODE_order : int
            the order of the differential equation that the network will solve. 
            The default value is None. If left as None then the network will not perform unsupervised training and
            will instead default to supervised training.
        q : float
            a hyper-parameter related to solving the Bernoulli family of differential equations.
        train_score : bool
            if True the network will return the train_score as well. This is for use in unsupervised 
            equations.
        epochs: int
            number of epochs to train (for use with unsupervised non-linear diffeqs)
        ode_coefs : list
            list of ODE coefficients
        random_sampling: bool
            if True then the network will perform random sampling, instead of uniform sampling, of time points.
            This makes the network considerably more powerful when solving diffeqs.
        sample_timepoints : int
            if the :attr:`random_sampling` argument is True then the network will sample this many timepoints.


        Returns
        -------
        int
            Description of return value

        """
        """

        TODO doctstring

        NLLLoss(),
        
        
        Arguments: TODO
            y: response matrix
            x: observer matrix
            burn in: obvious
            verbose:
        """
        # if verbose:
        #     if nl:
        #         print("nonlinear!")
        #     else:
        #         print("linear")

        if random_sampling:

            self._update = self._update_RandomSampling

        else:
            self._update = self._update_Iterative
        
        with torch.no_grad():
            
            non_assign_keys = ["X", "self", "y"]

            #assign all arguments input to the fit method as attributes
            for key, val in locals().items():
                if key not in non_assign_keys:
                    setattr(self, key, val)

            # if random_sampling and not self.ODE_order:
            #     assert False, "random sampling not implimented in supervised case"

            #self.reparam = reparam_f

            self.track_in_grad = False #track_in_grad

            if self.noise is not None:
                self.preactivation = self._preactivation_noise
            else:
                self.preactivation = self._preactivation_vanilla

            ########################### beta arguments are currently silenced #################
            ## these were used for Shaan's original paper idea
            #self.beta = beta
            # if beta is None:
            #     self.preactivation = self.preactivation_vanilla 
            # else:
            #     self.preactivation = self.preactivation_beta
            ########################### beta arguments are currently silenced #################
            

            #assert len(init_conditions) == ODE_order
            
            #SCALE = True
            if not self.ODE_order:
                SCALE = True
            else:
                SCALE = False
            
            #ensure that y is a 2-dimensional tensor with pre-specified arguments.
            if self.ODE_order: 

                #if we are solving differential equations unsupervised then use the 
                # `_train_states_unsupervised` method
                self.train_states = self._train_states_unsupervised
                
                assert init_conditions
                if len(init_conditions) > 1:
                    if type(init_conditions[1]) == list or type(init_conditions[1]) == np.ndarray:
                        #weird randomization condition
                        multiple_ICs = self.multiple_ICs = True
                        #print('randomizing init conds')
                        for i in range(len(init_conditions)):
                            init_conditions[i] = np.random.uniform(low  = init_conditions[i][0], 
                                                                   high = init_conditions[i][1])
                        init_conditions[0] = [init_conditions[i]]
                    elif type(init_conditions[0]) == list or type(init_conditions[0]) == np.ndarray:
                        multiple_ICs = self.multiple_ICs = True
                    else:
                        multiple_ICs = self.multiple_ICs = False
                elif type(init_conditions[0]) == list or type(init_conditions[0]) == np.ndarray:
                    multiple_ICs = self.multiple_ICs = True

                elif type(init_conditions[0]) in [float,  int, np.float64, np.float32, np.int32, np.float64]:
                    multiple_ICs = self.multiple_ICs = False
                else:
                    assert False, f'malformed ICs, either use a list, a float or an integer'
                init_conds = init_conditions
                
                
                # assert self.dt != None, "dt is None, bug found"
                # print("dt", self.dt)

                if self.dt == None:
                    dt_speculative = float(X[1]) - float(X[0])
                    if dt_speculative != 0:
                        self.dt = dt_speculative
                        print(f"assuming dt is {dt_speculative}, if this is incorrect please override dt.")

                if self.dt != None:
                    

                    start, stop = float(X[0]), float(X[-1])

                    if not random_sampling:
                        #there is a bug with the random sampling. Brute force attempt to fix it at first.

                        #here are lines I am commenting out
                        
                        self.alpha = self.leaking_rate[0] / self.dt
                        nsteps = int((stop - start) / self.dt)
                        X = torch.linspace(start, stop, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
                    else:
                        
                        X = torch.rand(sample_timepoints + 2) * (stop - start) + start

                        X = X.sort().values
                        x1, x2 = X[1:], X[:-1]
                        self.dt = x1-x2
                        
                        #if not hasattr(self, "leaking_rate_orig"):
                            
                        alpha = self.leaking_rate[0]**self.dt
                        self.leaking_rate_orig = deepcopy(self.leaking_rate)
                        self.leaking_rate = (1- alpha), alpha
                        self.alpha = self.leaking_rate[0] / self.dt
                        self.alpha = self.alpha.view(-1,1)
                    
                        
                        
                        X = X[:-1].reshape(-1,1)
                elif type(X) == type([]) and len(X) == 3:
                    x0, xf, nsteps = X #6*np.pi, 100
                    X = torch.linspace(x0, xf, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
                else:
                    assert False, f"Please input start, stop, dt, type(X): {type(X)}"

                if self.n_outputs is None:
                    if y is None:
                        self.n_outputs = X.shape[1]
                if y is None:
                    y = torch.ones((X.shape[0], self.n_outputs), **self.dev)

                
                
            else:
                #ensure that X is a two dimensional tensor, or if X is None declare a tensor.

                X = _check_x(X, y, self.dev, supervised = True)
                X.requires_grad_(False)
                y = _check_y(y, tensor_args = self.dev) 
                



                if random_sampling:
                    #X = torch.rand(sample_timepoints + 2)
                    idx = np.random.choice(range(X.shape[0]),sample_timepoints, replace = False)
                    idx.sort()

                    X = X[idx]
                    y = y[idx]

                    idx = torch.tensor(idx, **self.dev)


                    self._assign_random_sampling(idx)
                    assert False

                    #self.leaking_rate_orig = [new_predict_lr, 1-new_predict_lr]
                    #X = X[:-1]
                else:
                    self._assign_const_leaking_rate(X)
                
                
                    

                self.y_tr = y
                
                if SCALE:
                    y = self._scale(outputs=y, keep=True, normalize = self._normalize)    
                y.requires_grad_(False)

                self.lastoutput = y[ -1, :]
                self.multiple_ICs = multiple_ICs = False
                self.train_states = self._train_states_supervised

            self.unscaled_X = Parameter(X, requires_grad = self.track_in_grad)

            if self.unscaled_X.device != self.device:
                self.unscaled_X.data = self.unscaled_X.data.to(self.device)


            if self.unscaled_X.std() != 0:
                self.X = self.unscaled_X#.clone()

                if SCALE:
                    self.X.data = self._scale(inputs = self.unscaled_X, keep = True, normalize = False)#.clone()
            else:
                self._input_stds = None
                self._input_means = None
                self.X = self.unscaled_X


            ##at this point you can take a random sample? now it must happen sooner.
            self.X_extended = self._extend_X(self.X, self.random_sampling)

            if self.ODE_order and self.burn_in:
                x1, x2 = X[1:], X[:-1]
                
                y_extended = torch.ones((self.X_extended.shape[0], y.shape[1])).to(self.device)
            else:
                y_extended = y

            # if self.betas is None:
            #     self.betas = torch.ones_like(X[:,0].view(-1,1))

            start_index = 1 if self.feedback else 0 
            rows = X.shape[0] - start_index
            
            self.n_inputs = self.X.shape[1] 
            if not self.n_outputs and not self.ODE_order:
                self.n_outputs = y.shape[1]
            else:
                if not self.n_outputs:
                    assert False, 'you must enter n_outputs'

            if self.noise is not None:
                self.noise_z = torch.normal(0, 1, size = (self.n_nodes, self.X_extended.shape[0]), **self.dev) * self.noise
            
            self.lastinput = self.X[-1, :]
            
            start_index = 1 if self.feedback else 0 
            rows = X.shape[0] - start_index

            
            combined_weights = True

            if not self.LinOut:

                self.LinOut = Linear(self.n_nodes+1, self.n_outputs)
                #print("LinOut initial weights", self.LinOut.weight.shape)

                if combined_weights:
                    self.LinIn = Linear(2*self.n_inputs, self.n_nodes,  bias = False)
                    #self.LinFeedback = Linear(self.n_inputs, self.n_nodes, bias = False)
                    #self.LinFeedback = Linear(self.n_inputs, self.n_nodes, bias = False)

                else:
                    self.LinIn = Linear(self.n_inputs, self.n_nodes,  bias = False)
                    self.LinFeedback = Linear(self.n_inputs, self.n_nodes, bias = False)

                #self.LinIn = Linear(self.n_inputs, self.n_nodes,  bias = False)
                
                #self.LinIn.weight, self.LinFeedback.weight = self.set_Win()
                self.LinIn.weight = self._set_Win()


                # assert isinstance(n_inputs, int), "you must enter n_inputs. This is the number of input time series (int)"
                # assert isinstance(n_outputs, int), "you must enter n_outputs. This is the number of output time series (int)"

            #self.LinOutDE = Linear(self.n_nodes + 1, self.n_outputs)

            ################################################################################################
            #+++++++++++++++++++++++++++         FORWARD PASS AND SOLVE             ++++++++++++++++++++++++
            ################################################################################################
            with torch.set_grad_enabled(self.track_in_grad):
                self._freeze_weights()
                if out_weights:
                    #self.SOLVE = SOLVE = False
                    self.LinOut.weight =  Parameter(out_weights["weights"].to(self.device))
                    self.LinOut.bias = Parameter(out_weights["bias"].to(self.device)) 
                else:
                    try:
                        assert self.LinOut.weight.device == self.device
                    except:
                        self.LinOut.weight = Parameter(self.LinOut.weight.to(self.device))
                        self.LinOut.bias = Parameter(self.LinOut.bias.to(self.device))

                # CUSTOM_AUTOGRAD_F = False
                # if CUSTOM_AUTOGRAD_F:
                #     #for more information on custom pytorch autograd functions see the following link:
                #     #https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
                #     recurrence = Recurrence.apply

                #     self.states, self.states_dot, states_dot = recurrence(self.states, self, self.X_extended, y)
                # else:
                

                self.states = torch.zeros((1, self.n_nodes), **self.dev)
                

                #drop the first state and burned states
                self.states = self.train_states(self.X_extended, y_extended, self.states)
                
                if self.burn_in:
                    self.states = self.states[self.burn_in:]

                # calculate hidden state derivatives
                if self.ODE_order:
                    if self.feedback:
                        input_ = torch.hstack((self.X, y))
                        updates = self.LinIn(input_) + self.bias + self.LinRes(self.states)
                    else:
                        try:
                            updates = self.LinIn(self.X) + self.bias + self.LinRes(self.states)
                        except:
                            assert False, f"X_extended {self.X_extended.shape}, X {self.X.shape}, LinIn {self.LinIn.weight.shape}, self.LinRes {self.LinRes.weight.shape}, states {self.states.shape} "

                    self.states_dot = - self.alpha * self.states + self.alpha * self.activation_function(updates.T).T
                    self.states_dot = torch.cat((torch.ones_like(self.X), self.states_dot), axis = 1)
                    self.G = G = self.reparam_f(self.X)#, order = self.ODE_order)

                    # if self.ODE_order == 2:
                    #     self.states_dot2 = - self.alpha * self.states_dot + self.alpha * self.act_f_prime(updates) * (self.LinIn.weight.T + self.bias + self.LinRes(self.states_dot))
                    #     self.states_dot2 = torch.cat((torch.zeros_like(self.X), self.states_dot2), axis = 1)

                #append columns for the data:
                self.extended_states = torch.cat((self.X, self.states), axis = 1)
                #self.extended_states = torch.hstack((self.X, self.states))

                self.laststate = self.states[-1, :]

                del self.states

                #add rows corresponding to bias to the states 
                self.sb = states_with_bias = torch.cat((torch.ones_like(self.extended_states[:,0].view(-1,1)), self.extended_states), axis = 1)

                if self.ODE_order:
                    self.sb1 = states_dot_with_bias = torch.cat((torch.zeros_like(self.states_dot[:,0].view(-1,1)), self.states_dot), axis = 1)

                    # do the same for the second derivatives
                    # if self.ODE_order == 2:
                    #     self.sb2 = states_dot2_with_bias = torch.cat((torch.zeros_like(self.states_dot2[:,0].view(-1,1)), self.states_dot2), axis = 1)
                    g, g_dot = G
                    self.g = g

                    self.init_conds = init_conditions.copy()
                    ode_coefs = _convert_ode_coefs(self.ode_coefs, self.X)

                    self.force = force
                    self.force_t = self.force(self.X)



                #self.laststate = self.extended_states[-1, 1:]
                
                with torch.no_grad():

                    # if not SOLVE and not self.ODE_order:
                    #     train_x = self.extended_states
                    #     ones_row = torch.ones( train_x.shape[0], 1, **self.dev)
                    #     train_x = torch.hstack((ones_row, train_x))

                    if SOLVE: #and not out_weights:
                        #print("SOLVING!")



                        #include everything after burn_in 
                        if not self.ODE_order:
                            return self._solve_supervised(y, return_states, SCALE)

                        bias = None

                        #print("ridge regularizing")
                        with torch.set_grad_enabled(False):

                            if self.ODE_order:
                                if multiple_ICs:
                                    #only implemented for first order

                                    try:
                                        self.As = As = [y0 * g.pow(0) for y0 in self.init_conds[0]]
                                    except:
                                        assert False, f'{self.init_conds}, {self.init_conds}'
                                    init_cond_list = self.init_conds
                                    #init_cond_list = [[y0, self.init_conds[1]] for y0 in self.init_conds[0]]
                                     
                                else:
                                    A = self.A = [init_conds[i] * g.pow(i) for i in range(self.ODE_order)] 


                                t = self.X


                                # if self.Hargun_changes:
                                #     H, H_dot = self.extended_states, self.states_dot #
                                # else:
                                H, H_dot = states_with_bias, states_dot_with_bias
                                
                                
                                self.gH = gH = g * H

                                self.gH_mu = self.gH.mean()
                                self.gH_dot = gH_dot =  g_dot * H +  g * H_dot

                                if eq_system:
                                    
                                    S, S_dot = self.gH, gH_dot

                                    if multiple_ICs:
                                        p0 = self.init_conds[1]
                                        ones_vec = torch.ones_like(X).T 
                                        p0_vec = ones_vec * p0
                                        self.Cxs = Cxs = []
                                        self.Cps = Cps = []
                                        for y0 in self.init_conds[0]:
                                            #print("y0", y0)
                                            y0_vec = ones_vec * y0
                                            Cx = -y0_vec @ S + p0_vec @ S_dot
                                            Cp = -y0_vec @ S_dot - p0_vec @ S
                                            Cxs.append(Cx)
                                            Cps.append(Cp)

                                    else:
                                        y0, p0 = self.init_conds
                                        ones_vec = torch.ones_like(X).T 

                                        y0_vec = ones_vec * y0
                                        p0_vec = ones_vec * p0

                                        self.Cx = Cx = -y0_vec @ S + p0_vec @ S_dot
                                        self.Cp = Cp = -y0_vec @ S_dot - p0_vec @ S

                                    sigma1 = S.T @ S + S_dot.T @ S_dot

                                    self.Sigma=Sigma = sigma1 + self.regularization * torch.eye(sigma1.shape[1], **self.dev)

                                    self.Delta=Delta = S.T @ S_dot - S_dot.T @ S

                                    #try:
                                    delta_inv = torch.pinverse(Delta)
                                    sigma_inv = torch.pinverse(Sigma)
                                    D_H = Sigma @ delta_inv + Delta @ sigma_inv
                                    self.Lam=Lam = torch.pinverse(D_H)
                                    
                                    #assert False, f'{Cx.shape}, {Cp.shape} {Lam.shape} {Sigma.shape} {Delta.shape}'
                                    if not multiple_ICs:

                                        self.Wy = Wy =  Lam.T @ (Cx @ delta_inv  + Cp @  sigma_inv).T/len(self.X)
                                        self.Wp = Wp =  Lam.T @ (Cp @ delta_inv  - Cx @  sigma_inv).T/len(self.X)
                                        
                                        self.weight = weight = torch.cat((Wy.view(-1,1), Wp.view(-1,1)), axis = 1)

                                        bias = weight[0]
                                        weight = weight[1:].view(self.n_outputs, -1)

                                    else:
                                        self.weights_list = []
                                        self.biases_list = []
                                        Wys, Wps =  [], []
                                        for i in range(len(Cps)):
                                            Cx, Cp, nX = Cxs[i], Cps[i], len(self.X)
                                            Wy = Lam.T @ (Cx @ delta_inv  + Cp @  sigma_inv).T/nX
                                            Wp = Lam.T @ (Cp @ delta_inv  - Cx @  sigma_inv).T/nX
                                            Wys.append(Wy)
                                            Wps.append(Wp)

                                            weight = torch.cat((Wy.view(-1,1), Wp.view(-1,1)), axis = 1)

                                            bias = weight[0]
                                            weight = weight[1:].view(self.n_outputs, -1)

                                            self.weights_list.append(weight)
                                            self.biases_list.append(bias)
                                        weights = self.weights_list
                                        biases = self.biases_list
                                    
                                else:

                                
                                    #if self.ODE_order == 1:

                                    #oscillator
                                    #########################################################
                                    
                                    #########################################################

                                    #nonlinear:
                                    #########################################################
                                    if multiple_ICs:
                                        if nl:
                                            self.y0s = y0s  = init_conds[0]
                                            p = self.ode_coefs[0];
                                            self.D_As = D_As = [y0*(p+y0*self.q) - self.force_t for y0 in y0s]
                                        else:
                                            D_As = [A*ode_coefs[0]- self.force_t for A in As]
                                    else:

                                        if nl:
                                            y0  = init_conds[0]
                                            p = self.ode_coefs[0];
                                            #q=0.5
                                            self.D_A = D_A =  y0*(p+y0*self.q) - self.force_t
                                        else:
                                            ######x###################################################
                                            #population:
                                            self.D_A = D_A = A[0] * ode_coefs[0] - self.force_t   
                                    
                                        
                                    # D_A = A[0] * ode_coefs[0] - force(t)                                  
                                    """
                                    try:
                                        D_A = A * ode_coefs[0] - force(t)
                                    except:
                                        assert False, f'{ode_coefs[0].shape} {self.X.shape} {A[0].shape} {force(t).shape}'
                                    """
                                    # elif self.ODE_order == 2:
                                    #     w= self.ode_coefs[0]
                                    #     y0, v0 = init_conds[0], init_conds[1]
                                    #     D_A = v0 *(g_dot2 + w**2*g )+ w**2 * y0 -force(t)

                                    if nl:                                
                                        if not multiple_ICs:
                                            self.DH = DH =  gH_dot + p*gH + 2*self.q*y0*gH 
                                        else:
                                            self.DHs = DHs = [gH_dot + p*gH + 2*self.q*y0*gH for y0 in y0s]
                                    else:
                                        #assert False
                                        self.DH = DH = ode_coefs[0] * gH + ode_coefs[1] * gH_dot   

                                    # if self.ODE_order == 2:

                                    #     H_dot2 = states_dot2_with_bias
                                    #     DH = 2 * H *(g_dot ** 2 + g*g_dot2) + g*(4*g_dot*H_dot + g*H_dot2)  + w**2 * g**2*H
                                    #     term1 = 2*H*g_dot**2
                                    #     term2 = 2*g*(2*g_dot*H_dot + H*g_dot2)
                                    #     term3 = g**2*(w**2*H + H_dot2)
                                    #     DH = term1 +  term2 + term3
                                    #there will always be the same number of initial conditions as the order of the equation.
                                    #D_A = G[0] + torch.sum([ G[i + 1] * condition for i, condition in enumerate(initial_conditions)])
                                    if not nl:
                                        #################
                                        #DH = DH[:, 1:] <-- silenced...
                                        #DH = self._center_H(inputs = DH, keep = True)


                                        #self.extended_states = self.extended_states - self.extended_states.mean(axis = 0)
                                        #################
                                        DH1 = DH.T @ DH

                                        #xx, _ = self._center_data(DH, D_As[0])

                                        #if nl:
                                        #    DH1 = DH1 -2 * self.q* D_A.T * gH.T @ gH
                                        
                                        self.DH1 = DH1 = DH1 + self.regularization * torch.eye(DH1.shape[1], **self.dev)
                                        DHinv = torch.pinverse(DH1)
                                        self.DH2 = DH2 = DHinv @ DH.T
                                        if not multiple_ICs:
                                            weight = torch.matmul(-DH2, D_A)

                                            bias = weight[0]
                                            weight = weight[1:]
                                        else:
                                            #multiple ICs...

                                            weights, biases = [], []

                                            #DH = self._center_H(inputs = -DH, keep = True)

                                            for i, D_A in enumerate(D_As):


                                                #_  = self._center_H(outputs = D_A, keep = True)
                                                #bias = self._calc_bias(weight)


                                                weight = torch.matmul(-DH2, D_A)

                                                bias = weight[0]
                                                weight = weight[1:]


                                                biases.append(bias)
                                                weights.append(weight)

                                            self.biases_list = biases
                                            self.weights_list = weights

                                    else:
                                        ################################
                                        DH1s = [DH.T @ DH for DH in DHs]

                                        #gH_sq = gH.T @ gH 
                                        #DH1p2s = [-2 * self.q* D_A.T * gH_sq for D_A in D_As]
                                        nl_corrections = [2 * self.q* D_A.T * gH.T @ gH  for D_A in D_As]
                                        #self.DH1 -2 * self.q* D_A.T * gH.T @ gH

                                        self.DH1s = DH1s = [DH1s[i] - 0 for i, correction in enumerate(nl_corrections)]
                                        
                                        #DH1 = DH1 + self.regularization * torch.eye(DH1.shape[1], **self.dev)

                                        DHinvs = [torch.pinverse(DH1 + self.regularization * torch.eye(DH1s[0].shape[1], **self.dev))for DH1 in DH1s]
                                        self.DH2s = DH2s = [DHinv @ DHs[i].T for i, DHinv in enumerate(DHinvs)]

                                        weights, biases = [], []



                                        for i in range(len(D_As)):
                                            DH2 = DH2s[i]
                                            D_A = D_As[i]
                                            init_weight = torch.matmul(-DH2, D_A)

                                            biases.append(init_weight[0])
                                            weights.append(init_weight[1:])
                                        #assert False
                                        self.biases_list = biases
                                        self.weights_list = weights

                                    
                                        


                        #     elif not self.ODE_order:
                        #         ones_row = torch.ones( train_x.shape[0], 1, **self.dev)
                            
                        #         ridge_x = torch.matmul(train_x.T, train_x) + \
                        #                            self.regularization * torch.eye(train_x.shape[1], **self.dev)

                        #         ridge_y = torch.matmul(train_x.T, train_y)

                        #         ridge_x_inv = torch.pinverse(ridge_x)
                        #         weight = ridge_x_inv @ ridge_y

                        #         bias = weight[0]
                        #         weight = weight[1:]

                        # if not multiple_ICs:
                        #     self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                        #     self.LinOut.bias = Parameter(bias.view(1, self.n_outputs))

            if multiple_ICs:
                if not backprop_f:
                    init_conds_clone = init_conditions.copy()
                    ys = []
                    ydots = []
                    scores = []
                    last_outputs = []
                    #init_conds_clone = init_conds.copy()
                    for i, weight in enumerate(self.weights_list):
                        #print("w", i)
                        # if combined_weights:
                        #     self.LinOut.weight = Parameter(weight.view(self.n_outputs + 1, -1))
                        # else:
                        self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                        self.LinOut.bias = Parameter(biases[i].view(1, self.n_outputs))
                        if SOLVE:
                            self.LinOut.weight.requires_grad_(False)
                            self.LinOut.bias.requires_grad_(False)
                        

                        self.init_conds[0] = float(init_conds_clone[0][i])

                        N = self.LinOut(self.extended_states)
                        N_dot = self._calc_Ndot(self.states_dot, cutoff = False)
                        yfit = g*N
                        
                        if not eq_system:
                            yfit[:, 0] = yfit[:,0] + init_conds[0][i]
                        else:
                            
                            yfit[:, 0] = yfit[:,0] + init_conds[0][i]
                            for j, cond in enumerate(1, init_conds):
                                yfit[:, j] = yfit[:,j] + cond
                        
                        if train_score:
                            ydot = g_dot * N +  g * N_dot
                            ydots.append(ydot)
                            score = ODE_criterion(X, yfit.data, ydot.data, self.LinOut.weight.data, 
                                                    ode_coefs = ode_coefs, init_conds = init_cond_list, 
                                                    enet_strength = self.enet_strength, enet_alpha = self.enet_alpha,
                                                    force_t = self.force_t)
                            scores.append(score)

                        last_outputs.append(y[-1, :])
                        ys.append(yfit)

                    self.lastoutput = y[-1, :]#.clone()
                    self.init_conds = init_conditions
                    if train_score:
                        return {"scores" : scores, 
                                "weights": weights, 
                                "biases" : biases,
                                "ys"     : ys,
                                "ydots"  : ydots}
                    else:
                        return ys, ydots
                else:

                    # if self.parallel_backprop:

                    # results = ray.get([execute_objective.remote(parallel_args_id, cv_samples, parameter_lst[i], i) for i in range(num_processes)])
                    # else:
                    gd_weights = []
                    gd_biases = []
                    ys = []
                    ydots =[]
                    scores = []
                    Ls = []
                    init_conds_clone = init_conditions.copy()
                    if not SOLVE:
                        orig_weights = self.LinOut.weight#.clone()
                        orig_bias = self.LinOut.bias#.clone()

                    self.parallel_backprop = False

                    weight_dicts = []
                    if self.parallel_backprop:
                        new_out_W = Linear(self.LinOut.weight.shape[1], self.n_outputs)

                        if SOLVE:
                            new_out_W.weight = Parameter(self.weights_list[i].view(self.n_outputs, -1)).requires_grad_(True)
                            new_out_W.bias = Parameter(self.biases_list[i].view(1, self.n_outputs)).requires_grad_(True)
                        else:
                            try:

                                new_out_W.weight = Parameter(orig_weights.view(self.n_outputs, -1)).requires_grad_(True)
                                new_out_W.bias = Parameter(orig_bias.view(1, self.n_outputs)).requires_grad_(True)
                            except:
                                new_out_W.weight = Parameter(orig_weights.reshape(self.n_outputs, -1)).requires_grad_(True)
                                new_out_W.bias = Parameter(orig_bias.reshape(1, self.n_outputs)).requires_grad_(True)



                        data2save = {#"rc" : self, 
                                     "custom_loss" : self.ODE_criterion, 
                                     "epochs" : self.epochs,
                                     "New_X" : self.extended_states.detach(),
                                     "states_dot": self.states_dot.detach().requires_grad_(False),
                                     #"orig_bias" : orig_bias,
                                     #"orig_weights" : orig_weights,
                                     "out_W" : new_out_W,
                                     "force_t" : self.force_t,
                                     "criterion" : torch.nn.MSELoss(),
                                     #"optimizer" : optim.Adam(      self.parameters(), lr = 0.05),
                                     "t"  : self.X,
                                     "G" : self.G,
                                     "gamma" : self.gamma,
                                     "gamma_cyclic" : self.gamma_cyclic,
                                     #"parameters" : self.parameters(),
                                     "spikethreshold": self.spikethreshold,
                                     "ode_coefs" : self.ode_coefs,
                                     "enet_alpha" : self.enet_alpha,
                                     "enet_strength" : self.enet_strength,
                                     "init_conds" : self.init_conds, 
                                     "backprop_args" : backprop_args
                                     }

                        self_id = ray.put(data2save)

                        weight_dicts = ray.get([execute_backprop.remote(self_id, y0) for y0 in init_conds_clone[0]])
                    
                    else:
                        for i, y0 in enumerate(init_conds_clone[0]):
                            #print("w", i)
                            if SOLVE:
                                self.LinOut.weight = Parameter(self.weights_list[i].view(self.n_outputs, -1)).requires_grad_(True)
                                self.LinOut.bias = Parameter(self.biases_list[i].view(1, self.n_outputs)).requires_grad_(True)
                            else:
                                self.LinOut.weight = Parameter(orig_weights.view(self.n_outputs, -1))
                                self.LinOut.bias = Parameter(orig_bias.view(1, self.n_outputs))
                            self.init_conds[0] = float(y0)
                            #print(self.init_conds[0])
                            #breakpoint()
                            with torch.enable_grad():
                                weight_dict = backprop_f(self, force_t = self.force_t, custom_loss = ODE_criterion, epochs = epochs)
                            weight_dicts.append(weight_dict)
                    last_outputs = []
                    for weight_dict in weight_dicts:
                        score=weight_dict["best_score"]
                        y = weight_dict["y"]
                        ydot = weight_dict["ydot"]
                        loss, gd_weight, gd_bias = weight_dict["loss"]["loss_history"], weight_dict["weights"],  weight_dict["bias"]
                        scores.append(score)
                        ys.append(y)
                        ydots.append(ydot)
                        gd_weights.append(gd_weight)
                        gd_biases.append(gd_bias)
                        Ls.append(loss)
                        last_outputs.append(y[-1, :])

                    self.init_conds = init_conditions


                    self.lastoutput = y[-1, :]

                    self.weights_list = gd_weights
                    self.biases_list = gd_biases
                    if train_score:
                        return {"scores" : scores, 
                                "weights": gd_weights, 
                                "biases" : gd_biases,
                                "ys"     : ys,
                                "ydots"  : ydots,
                                "losses" : Ls}
                    else:
                        return ys, ydots

                        #{"weights": best_weight, "bias" : best_bias, "loss" : backprop_args, "ydot" : ydot, "y" : y}


            else:

                # self.biases_list = [bias]
                # self.weights_list = [weight]

                self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                self.LinOut.bias = Parameter(bias.view(1, self.n_outputs))

                self.N = self.LinOut(self.extended_states)
                
                # Store last y value as starting value for predictions
                self.lastoutput = y[-1, :]

                if self.ODE_order >= 1:
                    #calc Ndot just uses the weights
                    #self.states_dot @ self.LinOut.weight
                    # if not SOLVE:
                    #     ones_row = torch.ones( X.shape[0], 1, **self.dev)
                    #     #print(f'adjusting extended_states, shape before: {self.extended_states.shape}')
                    #     self.extended_states = torch.hstack((ones_row, self.extended_states))
                    #     #print(f'adjusting extended_states, shape after: {self.extended_states.shape}')
                        
                    #     zeros_row = torch.zeros( X.shape[0], 1, **self.dev)
                    #     self.states_dot = torch.hstack((zeros_row, self.states_dot))
                    
                    N_dot = self._calc_Ndot(self.states_dot, cutoff = False)
                    #1st order
                    # self.ydot = g_dot * self.N +  g * N_dot
                    #2nd order
                    #if self.ODE_order == 1:
                        
                    if not eq_system:
                        self.yfit = init_conds[0] + g.pow(1) * self.N
                        self.lastoutput = self.yfit[-1, :]
                        self.ydot = g_dot * self.N +  g * N_dot
                    else:

                        self.yfit = g*self.N
                        self.lastoutput = self.yfit[-1, :]
                        
                        for i, cond in enumerate(self.init_conds):
                            self.yfit[:, i] = self.yfit[:,i] + cond

                        self.ydot = g_dot * self.N +  g * N_dot

                    return {#"scores" : scores, 
                                "weight": self.LinOut.weight.data, 
                                "bias" : self.LinOut.bias.data,
                                "y"     : self.yfit,
                                "ydot"  : self.ydot}

                    if train_score:
                        return ODE_criterion(X, self.yfit.data, self.ydot.data, self.LinOut.weight.data, 
                                                ode_coefs = ode_coefs, init_conds = self.init_conds, 
                                                enet_strength = self.enet_strength, enet_alpha = self.enet_alpha,
                                                force_t = self.force_t)
                    else:
                        return self.yfit, self.ydot

            # if self.ODE_order >= 2:
            #     v0 = self.init_conds[1]
            #     self.ydot =  g_dot*(v0+2*g*self.N) + g**2*N_dot 

            #     #self.ydot2 = gH_dot2[:,1:] @ self.LinOut.weight
            #     N_dot2 = self.states_dot2 @ self.LinOut.weight.T 
            #     term2_1 = 4*g*g_dot*N_dot
            #     term2_2 = v0*g_dot2 
            #     term2_3 = 2*self.N*(g_dot**2 + g*g_dot2)
            #     term2_4 = g**2*N_dot2
            #     self.ydot2 = term2_1 +term2_2 + term2_3 + term2_4
            #     self.yfit = init_conds[0] + init_conds[1] * g + g.pow(self.ODE_order) * self.N
            #     self.lastoutput = self.yfit[-1, :]
            #     if train_score:
            #         return ODE_criterion(X= X, 
            #                              y = self.yfit.data, 
            #                              ydot = self.ydot.data,
            #                              ydot2 = self.ydot2.data, 
            #                              out_weights = self.LinOut.weight.data, 
            #                              ode_coefs = ode_coefs,
            #                              init_conds = self.init_conds,
            #                              enet_strength = self.enet_strength, 
            #                              enet_alpha = self.enet_alpha)
            #     return self.yfit, self.ydot, self.ydot2
            

            # if not ODE_order and burn_in:
            #     self.N = self.N[self.burn_in:,:]
            #     self.N = self.N.view(-1, self.n_outputs)
            #     self.X = self.X[self.burn_in:]
            
            # # Return all data for computation or visualization purposes (Note: these are normalized)
            # if return_states:
            #     return extended_states, (y[1:,:] if self.feedback else y), burn_in
            # else:
            #     self.yfit = self.LinOut(self.extended_states)
            #     if SCALE:   
            #         self.yfit = self._output_stds * self.yfit + self._output_means
            #     return self.yfit    

    def error(self, predicted, target, method='nmse', alpha=1.):
        """
        Evaluates the error between predictions and target values.

        Suggested values of alpha (see the parameter description below for an explanation):

        .. list-table:: n vs alpha values
                :widths: 25 25
                :header-rows: 1

                * - n
                  - alpha
                * - 1
                  - 1.6
                * - 2
                  - 2.8
                * - 3
                  - 4.0
                * - 4
                  - 5.2
                * - 5
                  - 6.4
                * - 6
                  - 7.6

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
            Alpha coefficient to scale the tanh error transformation: ``alpha * tanh{(1 / alpha) * error}``.
            This squeezes errors onto the interval [0, alpha].
            Default is 1. Suggestions for squeezing errors > n * stddev of the original series
            (for tanh-nrmse, this is the point after which difference with y = x is larger than 50%,
            and squeezing kicks in). suggested n, alpha value pairs:

        Returns
        -------
        error : float
            The error as evaluated with the metric chosen above
        """
        errors = predicted - target

        # Adjust for NaN and np.inf in predictions (unstable solution)
        #if not torch.all(torch.isfinite(predicted)):
        #    # print("Warning: some predicted values are not finite")
        #    errors = torch.inf
        
        def nmse(y, yhat):
            """
            normalized mean square error
            """
            return ((torch.sum(torch.square(y - yhat)) / torch.sum(torch.square(y)))) / len(y.squeeze())

        # Compute mean error
        if type(method) != type("custom"):
            #assert self.custom_criterion, "You need to input the argument `custom criterion` with a proper torch loss function that takes `predicted` and `target` as input"
            try:
                error = method(self.X_test, target, predicted)
            except:
                error = method(target = target, predicted = predicted)

            """
            try:
                error = 
            except:
                if type(method) == type("custom"):
                    pass
                else:
                assert False, "bad scoring method, please enter a string or input a valid custom loss function"
            """
        elif method == 'mse':
            error = torch.mean(torch.square(errors))
        elif method == "combined":
            nmse = torch.mean(torch.square(errors)) / torch.square(target.squeeze().std())

            kl = torch.sigmoid(torch.exp(torch.nn.KLDivLoss(reduction= 'sum')(
                torch.softmax(predicted, dim = -1), 
                torch.softmax(target, dim = -1))))
            error = nmse + kl
            print('score', 'nmse', nmse, 'kl', kl, 'combined', error)
        elif method == "trivial_penalty":
            mse = torch.mean(torch.square(errors))
            penalty = torch.square((1/predicted).mean())
            error = mse + penalty
            print('score', 'mse', mse.data, 'penalty', penalty.data, 'combined', error.data)
        elif method == "smoothing_penalty":
            mse = torch.mean(torch.square(errors))
            penalty = torch.square(self.dydx2).mean()
            error = mse + 0.1 * penalty
            print('score', 'mse', nmse, 'penalty', penalty, 'combined', error)
        elif method == "combined_penalties":
            mse = torch.mean(torch.square(errors))
            #we should include hyper-parameters here.
            dxpenalty = torch.log(torch.abs(self.dydx2))
            dxpenalty_is_positive = (dxpenalty > 0)*1
            dxpenalty = dxpenalty * dxpenalty_is_positive
            dxpenalty = dxpenalty.mean()
            nullpenalty = torch.square((1/predicted).mean())
            error = mse + dxpenalty + nullpenalty
            print('score', 'mse', mse.data, 'dydx^2_penalty', dxpenalty.data, "penalty2", nullpenalty.data, 'combined', error.data)
        elif method == 'tanh':
            error = alpha * torch.tanh(torch.mean(torch.square(errors)) / alpha)  # To 'squeeze' errors onto the interval (0, 1)
        elif method == 'rmse':
            error = torch.sqrt(torch.mean(torch.square(errors)))
        elif method == 'nmse':
            error = torch.mean(torch.square(errors)) / torch.square(target.squeeze().std())#ddof=1))
        elif method == 'nrmse':
            error = torch.sqrt(torch.mean(torch.square(errors))) / target.flatten().std()#ddof=1)
        elif method == 'tanh-nrmse':
            nrmse = torch.sqrt(torch.mean(torch.square(errors))) / target.flatten().std(ddof=1)
            error = alpha * torch.tanh(nrmse / alpha)
        elif method == 'log':
            mse = torch.mean(torch.square(errors))
            error = torch.log(mse)
        elif method == 'log-tanh':
            nrmse = torch.sqrt(torch.mean(torch.square(errors))) / target.flatten().std(ddof=1)
            error = torch.log(alpha * torch.tanh((1. / alpha) * nrmse))
        else:
            raise ValueError('Scoring method not recognized')
        return error#.type(self.dtype)

    def test(self, y, X=None, y_start=None, steps_ahead=None, scoring_method='nmse', 
                  alpha=1., scale = False, criterion = None, reparam = None,
                  ODE_criterion = None): # beta = None
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

        ########################## betas are currently silenced ################
        #self.beta = beta
        ########################## betas are currently silenced ################


        if not self.ODE_order:

            y = _check_y(y, tensor_args = self.dev)
            X = _check_x(X , y, self.dev, supervised = True).requires_grad_(True)
            final_t =y.shape[0]
            self.y_val = y
            self.steps_ahead = steps_ahead
            self.leaking_rate = self.leaking_rate_orig
            self._assign_const_leaking_rate(X)
            
            
        else:
            if self.dt != None:
                

                start, stop = float(X[0]), float(X[-1])
                nsteps = int((stop - start) / self.dt)
                X = torch.linspace(start, stop, steps = nsteps, requires_grad=False).view(-1, 1)#.to(self.device)
            elif type(X) == type([]) and len(X) == 3:
                x0, xf, nsteps = X #6*np.pi, 100
                X = torch.linspace(x0, xf, steps = nsteps, requires_grad=False).view(-1, 1)#.to(self.device)
            final_t =X.shape[0]

            ode_coefs = _convert_ode_coefs(self.ode_coefs, X)

        X.requires_grad_(False)

        assert X.requires_grad == False#self.track_in_grad
        
        # Run prediction

        if steps_ahead is None:
            if not self.ODE_order:
                self._pred = self.predict(n_steps = y.shape[0], X=X, y_start=y_start, scale = True)
                
                score = self.error(predicted = self._pred, target = y, method = scoring_method, alpha=alpha)
                if self.id_ == None:
                    return score, self._pred.data #{"yhat": y_predicted.data, "ytest": y}, X[self.burn_in:]
                else:
                    
                    score.detach()
                    self._pred.detach()
                    return score.detach(), self._pred.detach(), self.id_
            else:
                val_force_t = self.force(X)
                assert not scale
                if not self.multiple_ICs:

                    returns = self.predict(n_steps = X.shape[0], X=X, y_start=y_start, scale = scale,
                                   continue_force = True)

                    if self.ODE_order == 1:
                        y_predicted, ydot = returns
                    elif self.ODE_order == 2:
                        y_predicted, ydot, ydot2 = returns

                    if self.ODE_order == 1:
                        score = self.ODE_criterion(X, y_predicted.data, ydot.data, 
                                                self.LinOut.weight.data, 
                                                ode_coefs = ode_coefs, 
                                                init_conds = self.init_conds,
                                                force_t = val_force_t,
                                                enet_alpha = self.enet_alpha, 
                                                enet_strength = self.enet_strength) 
                    # elif self.ODE_order == 2:
                    #     score = ODE_criterion(X, y_predicted.data, ydot.data, ydot2.data, self.LinOut.weight.data, ode_coefs = ode_coefs, init_conds = self.init_conds,
                    #                           enet_alpha = self.enet_alpha, enet_strength = self.enet_strength) #error(predicted = y_predicted, target = dy_dx_val, method = scoring_method, alpha=alpha)
                    # # else:
                    # #     assert False
                    # #     score = self.error(predicted = y_predicted, target = y, method = scoring_method, alpha=alpha)
                    # Return error

                    self._pred = y_predicted
                    #print("score", score.detach())
                    return score.detach(), y_predicted.detach(), self.id_

                else:
                    y_preds, ydots = self.predict(n_steps = X.shape[0], X=X, y_start=y_start, scale = scale,
                                   continue_force = True)
                    scores = []
                    for i, pred in enumerate(y_preds):
                        ydot = ydots[i]
                        if not self.eq_system:
                            
                            score = self.ODE_criterion(X, pred.data, ydot.data, 
                                                  self.LinOut.weight.data, 
                                                  ode_coefs = ode_coefs, 
                                                  init_conds = self.init_conds,
                                                  enet_alpha = self.enet_alpha, 
                                                  enet_strength = self.enet_strength,
                                                  force_t = val_force_t)
                            
                        else:
                            init_conds_system =[self.init_conds[0][i]] + self.init_conds[1:]
                            y0, p0 = init_conds_system
                            #ham0 = 
                            score = self.ODE_criterion(X, pred.data, ydot.data, 
                                                  self.LinOut.weight.data, 
                                                  ode_coefs = ode_coefs, 
                                                  init_conds = init_conds_system,
                                                  enet_alpha = self.enet_alpha, 
                                                  enet_strength = self.enet_strength,
                                                  #ham0 = (1/2)*p0**2 - 3*y0**2 + (21/4)*y0**4,
                                                  force_t = val_force_t)
                        scores.append(score)
                        #print("score", score.detach())
                    
                    return scores, y_preds, self.id_



                #y_predicted, ydot = self.reparam(self.y0, X, N, N_dot)
            #_printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            assert False, f'predict_stepwise not implimented'
            y_predicted = self.predict_stepwise(y, X, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]

    def predict(self, n_steps, X=None, y_start=None, continuation = True, scale = True, continue_force = True):
        """Predicts n values in advance.

        Prediction starts from the last state generated in training.

        Parameters
        ----------
        n_steps : int
            The number of steps to predict into the future (internally done in one step increments)
        x : numpy array or None
            If prediciton requires inputs, provide them here
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value dfrom training will be used

        Returns
        -------
        y_predicted : numpy array
            Array of n_step predictions

        """
        # Check if ESN has been trained
        assert self.lastoutput is not None, 'Error: ESN not trained yet'

        self.unscaled_Xte = X

        # Normalize the inputs (like was done in train)
        if not X is None:
            if scale and self.unscaled_X.std() != 0:
                self.X_val = Parameter(self._scale(inputs=self.unscaled_Xte, normalize = False))
            else:
                self.X_val = Parameter(X)

        if y_start is not None:
            continuation = False

        if self.noise is not None:
            self.noise_z = torch.normal(0, 1, size = (self.n_nodes, self.X_val.shape[0]), **self.dev) * self.noise

        # try:
        #     assert self.X_val.device == self.device, ""
            
        # except:
        #     self.X_val.data = self.X_val.to(self.device)

        self.X_val_extended =  self.X_val
        #assert False, f'X mean {self.X_val.mean()} std {self.X_val.std()}'
        # if not continue_force:
        #     if self.ODE_order:
        #         continuation = False
        dev = {"device" : self.device, "dtype" : self.dtype, "requires_grad": False}

        n_samples = self.X_val_extended.shape[0]

        # if y_start: #if not x is None:
        #     if scale:
        #         previous_y = self.scale(outputs=y_start)[0]
        #     else:
        #         previous_y = y_start[0]
        assert self.lastinput.device == self.device
        if continuation:
            #if self.ODE_order >=2:
            #    lasthdot2 = self.lasthdot2
            #lasthdot = self.lasthdot
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            #if self.ODE_order >=2:
            #    lasthdot2 = torch.zeros(self.n_nodes, **dev)
            #lasthdot = torch.zeros(self.n_nodes, **dev)
            laststate = torch.zeros(self.n_nodes, **dev)
            lastinput = torch.zeros(self.n_inputs, **dev).view(-1, self.n_inputs)
            lastoutput = torch.zeros(self.n_outputs, **dev)
        #if self.ODE_order:
        #    lastoutput = torch.zeros(self.n_outputs, **dev)


        if not y_start is None:

            lastoutput = self._scale(outputs=y_start, normalize = self._normalize)# self.scale(inputs=X)

        inputs = torch.vstack([lastinput, self.X_val_extended]).view(-1, self.X_val_extended.shape[1])
        
        states = torch.zeros((1, self.n_nodes), **dev)
        states[0,:] = laststate
        outputs = lastoutput.view(1, self.n_outputs)

        #assert False, f"last output {lastoutput}, ystart {y_start}"
        

        #dt = inputs[1,:] - inputs[0,:]
        with torch.no_grad():
            
            if self.ODE_order:
                states_dot = torch.zeros((1, self.n_nodes), **dev)
                #states, outputs = self.unsupervised_val_states(n_samples, inputs, states, outputs)
                states, outputs = self._unsupervised_val_states(n_samples, inputs, states, outputs)

            else:
                #states = torch.vstack((states, states))
                states, outputs = self._supervised_val_states(n_samples, inputs, states, outputs)

            #drop first state and first output (final train-set datapoints which were already used)
            self.val_states = states = states[1:]
            outputs = outputs[1:]

            
            # if self.burn_in:
            #    states = states[self.burn_in:]
            #    outputs = outputs[self.burn_in:]

            if not self.ODE_order:
                # try:
                #     if scale:
                yhat = self._descale(outputs = outputs, normalize = self._normalize).view(-1, self.n_outputs) 
                # except:
                #     yhat = outputs
                return yhat
            else:
                # calculate hidden state derivatives
                if not self.feedback:
                    updates = self.LinIn(self.X_val) + self.bias + self.LinRes(states)
                else:
                    input_ = torch.hstack((self.X_val, outputs))
                    updates = self.LinIn(input_) + self.bias + self.LinRes(states)

                states = torch.cat((self.X_val, states), axis = 1)
                time = self.X_val
                states_dot = - self.alpha * states[:,1:] + self.alpha * self.activation_function(updates)
                # if self.ODE_order == 2:
                #     states_dot2 = - self.alpha * states_dot + self.alpha * self.act_f_prime(updates) * (self.LinIn.weight.T + self.bias + self.LinRes(states_dot))
                #     states_dot2 = torch.cat((torch.zeros_like(self.X_val), states_dot2), axis = 1)
                states_dot = torch.cat((torch.ones_like(self.X_val), states_dot), axis = 1)
                assert states.shape == states_dot.shape

                G = self.reparam_f(self.X_val, order = self.ODE_order)
                g, g_dot = G

                self.val_states_dict = {"s" : states, "s1" : states_dot, "G" :  G}


                # if self.ODE_order == 2:
                #     #derivative of  g_dot * states_with_bias
                #     #gH_dot2_p1 =  g_dot * states_dot  + g_dot2 * states

                #     #derivative of  g * states_dot_with_bias
                #     #gH_dot2_p2 =  g * states_dot2  + g_dot * states_dot
                    
                #     #gH_dot2 = gH_dot2_p1 + gH_dot2_p2

                #     #ydot2 = self.LinOut(gH_dot2)

                #     N_dot2 = states_dot2 @ self.LinOut.weight.T #self.calc_Ndot(states_dot2, cutoff = False)
                #     ydot2 = 4*g*g_dot*N_dot + self.init_conds[1]*g_dot2
                #assert False, f'o {outputs.shape} t {time.shape} dN {N_dot.shape}'
                assert type(self.reparam_f) != type(None), "you must input a reparam function with ODE"
                
                if not self.multiple_ICs:

                    A = self.init_conds[0] * g.pow(self.ODE_order)

                    #the code may be useful for solving higher order:
                    ##########################
                    #A = [ self.init_conds[i] * g.pow(i) for i in range(self.ODE_order)]
                    # for i in range(self.ODE_order):
                    #     A_i = self.init_conds[i] * g.pow(i) #self.ode_coefs[i] * 
                    #     if not i:
                    #         A = A_i # + ode_coefs[1] * v0 * g.pow(1) + ... + ode_coefs[m] * accel_0 * g.pow(m)
                    #     else:
                    #         A = A + A_i
                else:
                    As = [y0 * g.pow(0) for y0 in self.init_conds[0]]

                if not self.multiple_ICs:
                    if not self.eq_system:
                        N = self.LinOut(states)
                        N_dot = self._calc_Ndot(states_dot, cutoff = False)

                        y = A + g * N

                        for i, cond in enumerate(range(y.shape[1])):
                            

                            y[:, i] = y[:,i] + self.init_conds[i]

                        ydot = g_dot * N +  g * N_dot
                        
                        #self.N = self.N.view(-1, self.n_outputs)
                        #gH = g * states
                        #gH_dot =  g_dot * states  +  g * states_dot

                        
                        #elf.yfit = init_conds[0] + g.pow(1) * self.N
                        #self.ydot = g_dot * self.N +  g * N_dot
                    else:
                        y = g * self.LinOut(states)
                        ydot = g_dot * N + N_dot * g
                        for i, cond in enumerate(self.init_conds):

                            y[:, i] = y[:,i] + cond
                    return y, ydot
                else:
                    if not self.eq_system:
                        ys = []
                        ydots = []
                        for i, weight in enumerate(self.weights_list):
                            #print("w", i)
                            self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                            self.LinOut.bias = Parameter(self.biases_list[i].view(1, self.n_outputs))

                            N = self.LinOut(states)
                            N_dot = self._calc_Ndot(states_dot, cutoff = False)
                            yfit = g*N


                            yfit[:, 0] = yfit[:,0] + self.init_conds[0][i]
                            
                            ydot = g_dot * N +  g * N_dot

                            ys.append(yfit)
                            ydots.append(ydot)

                        return ys, ydots
                    else:
                        ys = []
                        ydots = []
                        #loop
                        for i, weight in enumerate(self.weights_list):
                            self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                            self.LinOut.bias = Parameter(self.biases_list[i].view(1, self.n_outputs))

                            N = self.LinOut(states)
                            N_dot = self._calc_Ndot(states_dot, cutoff = False)

                            #reparameterize
                            y = g*N
                            ydot = g_dot * N + N_dot * g

                            #add initial conditions
                            y[:, 0] = y[:,0] + self.init_conds[0][i]
                            for j in range(1, y.shape[1]):
                                y[:, j] = y[:,j] + self.init_conds[j]
                            
                            ys.append(y)
                            ydots.append(ydot)

                        return ys, ydots

                
                # if self.ODE_order == 2:
                #     y0, v0 = self.init_conds
                #     ydot =  g_dot*(v0+2*g*N) + g**2*N_dot 

                #     #self.ydot2 = gH_dot2[:,1:] @ self.LinOut.weight
                #     N_dot2 = states_dot2 @ self.LinOut.weight.T 
                #     term2_1 = 4*g*g_dot*N_dot
                #     term2_2 = v0*g_dot2 
                #     term2_3 = 2*N*(g_dot**2 + g*g_dot2)
                #     term2_4 = g**2*N_dot2
                #     ydot2 = term2_1 +term2_2 + term2_3 + term2_4
                #     y = y0 + v0 * g + g.pow(self.ODE_order) * N
                #     #y = A + g**2 * N
                #     return y, ydot, ydot2
                
        #https://towardsdatascience.com/in-place-operations-in-pytorch-f91d493e970e

    def plot_prediction(self, axis_label_fontsize = None, 
                              fig = None, 
                              gt_tr_override = None, 
                              gt_te_override = None, 
                              lw_vert = None,
                              prep = True, 
                              tick_fontsize = None,
                              ylabel = None) -> None:
        """plots the RC predictions

        Arguments
        ---------
        gt_tr_override: if you want to calculate residuals from a non-noisy real training set,
            residuals will be calculated from it not say noisy inputs

        gt_te_override: if you want to calculate residuals from a non-noisy real validation set,
            residuals will be calculated from it not say noisy inputs
        fig: a matplotlib figure to plot on. 

        ylabel: the users desired ylabel
        """

        if not fig:
            plt.figure(figsize = (16, 4))

        lw = 3
        
        if prep:
            self._plot_prep(gt_tr_override, gt_te_override)

        #do you want discrete time steps or the actual value of the input?
        try:
            #if you want actual value
            input_train, input_test = self.t_tr, self.t_te  
        except:
            #if you want the discrete indices/ don't supply the input.
            #assert False, "fix this later"
            input_train, input_test = self.tr_idx, self.te_idx


        if not self._override:
            pred_alpha = 0.4
            gt_alpha = 1
            plt.plot(input_train, self.yfit,  alpha = pred_alpha, linewidth = lw+2, #color = "blue",
                label = "train")

            plt.plot(input_test, self._pred, alpha = pred_alpha, linewidth = lw+2, #color = "red", 
                    label = "test")
            try:
                input_train, input_test = list(input_train.ravel()), list(input_test.ravel())
            except:
                pass

            plt.plot(input_train + input_test,
                     np.concatenate((self._gt_tr, self._gt_te), axis = 0),
                     '--',
                     color = "black",
                     alpha = gt_alpha,
                     linewidth = lw-1,

                     label = "ground truth")

        else:
            pred_alpha = 0.9
            gt_alpha = 0.3
            plt.plot(input_train + input_test, #self.tr_idx + self.te_idx,
                     np.concatenate((self.y_tr, self.y_val), axis = 0),
                     '--',
                     color = "black",
                     alpha = gt_alpha,
                     linewidth = lw-1,
                     label = "ground truth")

        
            plt.plot(input_train, self.yfit,  alpha = pred_alpha, linewidth = lw, #color = "blue",
                label = "train")
            plt.plot(input_test, self._pred, alpha = pred_alpha, linewidth = lw, #color = "red", 
                    label = "test")

        plt.axvline(input_train[-1], linestyle = ':', color = 'darkblue',  linewidth = lw_vert)
        
        if ylabel is not None:
            plt.ylabel(ylabel, fontdict = {"fontsize": axis_label_fontsize})
        #plt.legend()
        if tick_fontsize is not None:
            for axis in ['x', 'y']:
                plt.tick_params(axis=axis, labelsize=tick_fontsize)

    def plot_residuals(self, 
                       axis_label_fontsize : int = None, 
                       fig = None,
                       gt_tr_override = None, 
                       gt_te_override = None, 
                       lw_vert = None, 
                       prep : bool = True,
                       tick_fontsize : int = None,
                       ylabel = None
                       ) -> None:
        """
        Residuals plot

        Extended description of function.

        Parameters
        ----------
        axis_label_fontsize : int
            fontsize for axis labels
        fig  : matplotlib.figure.Figure
            Figure to use to plot. If None, a Figure will be generated. 
            matplotlib.figure.Figure: The top level container for all the plot elements.
        gt_tr_override : dtype
            Description of arg3
        gt_te_override : dtype
            Description of arg4
        lw_vert : dtype
            Desc
        prep : bool
            Desc
        tick_fontsize: int
            Desc
        ylabel: dtype
            Desc

        Returns
        -------
        None

        """
        if prep:
            self._plot_prep(gt_tr_override, gt_te_override)
        if not fig:
            plt.figure(figsize = (16, 4))

        #do you want discrete time steps or the actual value of the input?
        try:
            #if you want actual value
            input_train, input_test = self.t_tr, self.t_te  
        except:
            #if you want the discrete indices/ don't supply the input.
            #assert False, "fix this later"
            input_train, input_test = self.tr_idx, self.te_idx

        
        plt.plot(input_train, self.tr_resids)
        plt.plot(input_test, self.te_resids)
        
        plt.axvline(input_train[-1], linestyle = ':', color = 'darkblue',  linewidth = lw_vert)
        if ylabel is not None:
            plt.ylabel(ylabel, fontdict = {"fontsize": axis_label_fontsize})
        plt.xlabel(r"$t$", fontdict = {"fontsize": axis_label_fontsize})
        plt.yscale("log")


        if tick_fontsize is not None:
            for axis in ['x', 'y']:
                plt.tick_params(axis=axis, labelsize=tick_fontsize)

    def combined_plot(self, t_tr = None, 
                            t_te = None, 
                            fig = None, 
                            gt_tr_override = None, 
                            gt_te_override = None,  
                            axis_label_fontsize = 29,
                            lw_vert = 2, 
                            tight_layout_args = {'w_pad' : 0.00, 'h_pad' : 0.1, 'pad' : 0.01 }, 
                            ylabel_pred = None, 
                            tick_fontsize= 27,
                            ylabel_resid = r'$MSE$',
                            grid_spec_x = 7,
                            labelsize  = 20,
                            resid_blocks = 4) -> None:
        """
        Plots both residuals and the prediction.

        Extended description of function.

        tight_layout_args: must be a dictionary,
            for example: {'pad'=0.4, 'w_pad'=0.5, 'h_pad'=1.0}
            #https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html

        Parameters
        ----------
        axis_label_fontsize : int
            fontsize for axis labels
        fig  : matplotlib.figure.Figure
            Figure to use to plot. If None, a Figure will be generated. 
            matplotlib.figure.Figure: The top level container for all the plot elements.
        grid_spec_x : dtype
            Desc
        gt_tr_override : dtype
            Desc
        gt_te_override  : dtype
            Desc
        labelsize : int
            font size for labels
        lw_vert  : dtype
            Desc
        resid_blocks : int
            Desc
        tick_fontsize : dtype
            Desc
        tight_layout_args  : dtype
            Desc
        t_tr : dtype
            Description of arg1
        t_te : dtype
            Description of arg2
        ylabel_pred : dtype
            Desc
        ylabel_resid : dtype
            Desc

        Returns
        -------
        None

        """
        #assign the time train and time test, if they exist.
        if t_tr is not None:
            self.t_tr = t_tr
        if t_te is not None:
            self.t_te = t_te

        self._plot_prep(gt_tr_override, gt_te_override)
        if not fig:
            fig = plt.figure(figsize = (16,7))
            g = grid_spec_x
            gs1 = gridspec.GridSpec(g, g);
        ax = plt.subplot(gs1[:resid_blocks, :])

        self.plot_prediction(gt_tr_override = gt_tr_override, 
                             gt_te_override = gt_te_override, 
                             fig = 1, 
                             prep = True, 
                             axis_label_fontsize = axis_label_fontsize, 
                             tick_fontsize = tick_fontsize,
                             lw_vert = lw_vert, 
                             ylabel = ylabel_pred)

        ax.tick_params(labelbottom=False, labelsize = tick_fontsize)

        #[ (plt.sca(ax[i]), plt.xticks(fontsize=tick_fontsize), plt.yticks(fontsize=tick_fontsize)) for i in range(3)]

        ax = plt.subplot( gs1[resid_blocks:, :] )
        
        self.plot_residuals(fig = 1,
                            #prep = False, 
                            lw_vert = lw_vert, 
                            ylabel = ylabel_resid,
                            axis_label_fontsize = axis_label_fontsize, 
                            tick_fontsize = tick_fontsize)
        
        if tight_layout_args is not None:
            plt.tight_layout(**tight_layout_args)
        else:
            plt.tight_layout()

    def _assign_random_sampling(self, X):
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
        x1, x2 = X[1:], X[:-1]
        self.dt = x1-x2
        #print("dt", self.dt)
        

        alpha = self.leaking_rate[0]**self.dt
        #breakpoint()

        
        self.leaking_rate =  alpha, (1- alpha)

        self.alpha = self.leaking_rate[0] / self.dt
        self.alpha = self.alpha.view(-1,1)

    def _assign_const_leaking_rate(self, X):
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
        #self.leaking_rate_orig = self.leaking_rate.copy()
        pass
        # ones= torch.ones((X.shape[0], 1)).to(self.device)
        # try:
        #     self.leaking_rate[0] = self.leaking_rate_orig[0] * ones
        #     self.leaking_rate[1] = self.leaking_rate_orig[1] * ones
        # except:
        #     self.leaking_rate[0] = float(self.leaking_rate_orig[0][0]) * ones
        #     self.leaking_rate[1] = float(self.leaking_rate_orig[1][0]) * ones
    
    #from this point on are internal methods
    def _scale(self, inputs=None, outputs=None, keep=False, normalize = False):
        """Normalizes array by column (along rows) and stores mean and standard devation.

        Set `store` to True if you want to retain means and stds for denormalization later.

        .. warning::
            :meth:`_scale` is an internal method. As is standard, methods not designed for the user
            begin with the `_` character.


        Parameters
        ----------
        inputs : array or None
            Input matrix that is to be normalized
        outputs : array or No ne 
        no_grads            Output column vector that is to be normalized
        keep : bool
            Stores the normalization transformation in the object to denormalize later

        Returns
        -------
        transformed : tuple or array
            Returns tuple of every normalized array. In case only one object is to be returned the tuple will be
            unpacked before returning

        """
        self.tanh_bound_limit = 1 #.9
        # Checks
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')

        # Storage for transformed variables
        transformed = []
        if not inputs is None:

            if normalize:
                if keep:
                    # Store for denormalization
                    self._input_mins = inputs.min(axis = 0).values
                    self._input_ranges = inputs.max(axis = 0).values - self._input_mins 

                # Transform
                if not self.ODE_order:
                    #normalize to between -0.5 and 0.5 for echostate purposes
                    if self.output_activation == "sin":
                        normalized = (((inputs - self._input_mins) / self._input_ranges) - 0.5)*2
                    else:
                        normalized = (((inputs - self._input_mins) / self._input_ranges) - 0.5)*self.tanh_bound_limit

                    preped = self.output_f_inv(normalized)

                    transformed.append( self._scale(inputs=preped, keep = keep))

                else:
                    assert False, "normalization not implimented for ODEs"
                    #transformed.append( inputs / self._input_stds)

            else:
                if keep:
                    # Store for destandardization
                    self._input_means = inputs.mean(axis=0)
                    self._input_stds = inputs.std(dim = 0)

                # Transform
                #if not self.ODE_order:
                transformed.append((inputs - self._input_means) / self._input_stds)
                #else: 
                #    transformed.append( inputs / self._input_stds)

        if not outputs is None:
            if normalize:
                if keep:
                    # Store for denormalization
                    self._output_mins = outputs.min(axis = 0).values
                    self._output_ranges = outputs.max(axis = 0).values - self._output_mins

                # Transform
                if not self.ODE_order:
                    #normalize to between -0.5 and 0.5 for echostate purposes
                    normalized = (((outputs - self._output_mins) / self._output_ranges) - 0.5)*self.tanh_bound_limit

                    preped = self.output_f_inv(normalized)

                    transformed.append( self._scale(outputs=preped, keep = keep))
                else: 
                    assert False, "normalization not implimented for ODEs"
                    #transformed.append( inputs / self._input_stds)
            else:
                if keep:
                    # Store for denormalization
                    self._output_means = outputs.mean(axis=0)
                    self._output_stds = outputs.std(dim = 0)#, ddof=1)

                # Transform
                if self.ODE_order:
                    transformed.append(outputs)
                else:
                    transformed.append((outputs - self._output_means) / self._output_stds)
                
                self._output_means = self._output_means
                self._output_stds = self._output_stds
        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

    def _descale(self, inputs=None, outputs=None, normalize = False):
        """
        Internal descaling function
        Denormalizes array by column (along rows) using stored mean and standard deviation.

        Parameters
        ----------
        inputs : array or None
            Any inputs that need to be transformed back to their original scales
        outputs : array or None
            Any output that need to be transformed back to their original scales
        normalize: bool
            Desc
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
        
        #for tensor in [train_x, train_y]:
        #     print('device',tensor.get_device())
        
        if normalize:
            if self.ODE_order:
                    assert False, "not implimented"
            if not inputs is None:
                assert False, f' there is no need to normalize the input, standardize instead'
                #transformed.append(( (inputs + 0.5) * self._input_ranges) + self._input_mins)

            if not outputs is None:

                destandardized = self._descale(outputs = outputs, normalize = False)

                if self.output_activation == "sin":
                    denormalized = ( self.output_f(destandardized/2) + 0.5) * self._output_ranges + self._output_mins

                else:

                    denormalized = ( self.output_f(destandardized/self.tanh_bound_limit) + 0.5) * self._output_ranges + self._output_mins

                transformed.append(denormalized)
        else:
            if not inputs is None:
                # if self.ODE_order:
                #     transformed.append(inputs * self._input_stds)
                # else:
                transformed.append((inputs * self._input_stds) + self._input_means)

            if not outputs is None:
                if self.ODE_order:
                    transformed.append(outputs)
                else:
                    transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

    def _plot_prep(self, gt_tr_override = None, gt_te_override = None):
        """
        internal plot preparation function

        Assigns various plotting attributes to the RcNetwork class

        Parameters
        ----------
        gt_tr_override : dtype
            If you are using noisy data this argument will allow you to plot the non-noisy ground truth
            on the training set
        gt_te_override : dtype
            If you are using noisy data this argument will allow you to plot the non-noisy ground truth
            on the test set
        Returns
        -------
        None

        """
        #for noise predictions, ground truth over-ride

        

        if gt_tr_override is None:
            self._gt_te = self.y_val
            self._gt_tr = self.y_tr
            self._override = False
        else:
            self._gt_te = gt_te_override
            self._gt_tr = gt_tr_override

            self._override = True
            #self._gt_tr = gt_tr_override #self._gt_tr = fit_args["y"]

        # if gt_te_override is None:
        #     self._gt_te = self.y_val
        # else:
        #     self._gt_te = gt_te_override

        self._len_te = len(self._pred)
        self._len_tr = len(self.yfit)

        self.te_idx = list(range(self._len_tr, self._len_te + self._len_tr))
            

        #save resids
        self.te_resids = (self._gt_te - self._pred)**2
        self.tr_resids = (self._gt_tr - self.yfit)**2

        self._xte = self.unscaled_Xte

        self._len_tr = len(self.yfit)
        self.tr_idx = list(range(self._len_tr))

    def _multiple_act_f(self, X):
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
        new_X = X
        for i, activation_function in enumerate(self._act_fs):

            mask = self._act_mask == i
            new_X[mask] = activation_function(new_X[mask])
        return new_X

    def _multiple_act_f_prime(self, X):
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
        new_X = X
        for i, activation_function in enumerate(self._act_f_primes):
            mask = self._act_masks[i]
            new_X[mask] = activation_function(new_X[mask])
        return new_X


    def _preactivation_vanilla(self, t, input_vector, recurrent_vec, bias): #, betas):
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
        return input_vector + recurrent_vec +  self.bias

    def _preactivation_noise(self, t, input_vector, recurrent_vec, bias):
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
        return input_vector + recurrent_vec +  self.bias + self.noise_z[:, t]

    def _train_state_feedback(self, t, X, state, y, output = False, retain_grad = False):
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
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.
        Returns:
            next_state: a torch.tensor which is the next hidden state
        """
        try:
            input_ = torch.hstack((X,y.squeeze()))
        except:
            breakpoint()

        input_vector = self.LinIn(input_)
        recurrent_vec = self.LinRes(state)
        preactivation = self.preactivation(t, input_vector, recurrent_vec, self.bias)#, self.beta)

        #feedback_vec = self.LinFeedback(y)

        #preactivation = preactivation + feedback_vec

        update = self.activation_function(preactivation)
        next_state = self._update(update, state, t)


        #self.leaking_rate[0][t-1] * update + self.leaking_rate[1][t-1] * state

        # if output:
        #     return next_state, self.LinOut(torch.cat([X, next_state], axis = 0).view(self.n_outputs,-1))
        #next_extended_state = torch.hstack((X, next_state)).view(1,-1)

        #breakpoint()
        #output = self.LinOut(next_extended_state)
        #assert False, f'{X.shape} {next_state.shape} {self.LinOut.weight.shape}, {output.shape} '
        
        return next_state, None#output #.view(self.n_outputs,-1))

    def _train_state_feedback_unsupervised(self, t, X, state, y, output = False, retain_grad = False):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.
        Returns:
            next_state: a torch.tensor which is the next hidden state
        """
        #try:
        input_ = torch.hstack((X,y.squeeze()))
        # except:
        #     breakpoint()

        input_vector = self.LinIn(input_)
        recurrent_vec = self.LinRes(state)
        preactivation = self.preactivation(t, input_vector, recurrent_vec, self.bias)#, self.beta)

        update = self.activation_function(preactivation)
        next_state = self._update(update, state, t)#self.leaking_rate[0][t-1] * update + self.leaking_rate[1][t-1] * state
        
        next_extended_state = torch.hstack((X, next_state)).view(1,-1)

        output = self.LinOut(next_extended_state)
        #assert False, f'{X.shape} {next_state.shape} {self.LinOut.weight.shape}, {output.shape} '
        
        return next_state, output

    def _train_state_vanilla(self, t, X, state, y, output = False, retain_grad = False):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.

        The function split makes sense for a speedup (remove the if statement)
        """
        #assert False
        input_vector = self.LinIn(X)
        recurrent_vec = self.LinRes(state)

        preactivation = self.preactivation(t, input_vector, recurrent_vec, self.bias)#, self.beta)

        update = self.activation_function(preactivation)
        next_state = self._update(update, state, t)#self.leaking_rate[0][t-1] * update + self.leaking_rate[1][t-1] * state
        # if output:
        #     return next_state, self.LinOut(torch.cat([X, next_state], axis = 0).view(self.n_outputs,-1))
        # else:
        return next_state, None

    def _train_state_vanilla_rs(self, t, X, state, y, output = False, retain_grad = False):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.

        The function split makes sense for a speedup (remove the if statement)
        """
        #assert False
        input_vector = self.LinIn(X)
        recurrent_vec = self.LinRes(state)

        preactivation = self.preactivation(t, input_vector, recurrent_vec, self.bias)#, self.beta)

        update = self.activation_function(preactivation)
        next_state = self._update(update, state, t)#self.leaking_rate[0][t-1] * update + self.leaking_rate[1][t-1] * state
        # if output:
        #     return next_state, self.LinOut(torch.cat([X, next_state], axis = 0).view(self.n_outputs,-1))
        # else:
        return next_state, None

    def _output_i(self, x, next_state):
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
        extended_state = torch.cat([x.view(-1,), next_state], axis = 0).view(1,-1)
        return self.LinOut(extended_state)

    def _supervised_val_states(self, n_samples, inputs, states, outputs):
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
        for t in range(n_samples):

            #assert False, f'inputs[t+1, :] :, {inputs[t+1, :]}'
            input_t, state_t, output_prev = inputs[t+1, :], states[t,:], outputs[t,:]

            state_t, _ = self.train_state(t, X = input_t, state = state_t, y = output_prev)

            states = torch.cat([states, state_t.view(-1, self.n_nodes)], axis = 0)

            output_t = self._output_i(input_t, state_t)

            outputs = torch.cat([outputs, output_t], axis = 0)
            
        return states, outputs

    def _unsupervised_val_states(self, n_samples, inputs, states, outputs):
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
        for t in range(n_samples):

            input_t, state_t, output_prev = inputs[t+1, :], states[t,:], outputs[t,:]

            state_t, output_t  = self.train_state(t, X = input_t, state = state_t, y = output_prev)
            states = torch.vstack((states, state_t.view(-1, self.n_nodes)))

            output_t = self._output_i(input_t, state_t)

            outputs = torch.vstack((outputs, output_t))

        return states, outputs

    def _update_RandomSampling(self, update, state, t):
        """Right here do the update if there is random sampling (leaking rate is a matrix, 2*n)"""
        next_state = self.leaking_rate[0][t-1] * update + self.leaking_rate[1][t-1] * state
        return next_state

    def _update_Iterative(self, update, state, t):
        #assert False, f"self.leaking_rate {self.leaking_rate}"
        """Right here do the update if there is no random sampling (leaking rate is a vector, 2*1)"""
        next_state = self.leaking_rate[0] * update + self.leaking_rate[1] * state
        return next_state


    def _calc_Ndot(self, states_dot, cutoff = True):
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
        """
        Parameters
        ----------
        cutoff: whether or not to cutoff
        """
        #if self.burn_in and cutoff:
        #    states_dot = torch.cat((states_dot[0,:].view(1,-1), states_dot[(self.burn_in + 1):,:]), axis = 0)
        #else:
        #    states_dot = states_dot
        dN_dx = states_dot @ self.LinOut.weight.T
        return dN_dx

    def _gen_discrete_weights(self, sigma, dim, connectivity):
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
        sparcity, c = (1 - connectivity), connectivity/2
        np_weights = np.random.choice([0, -sigma, sigma],  p = [sparcity, c, c], size = dim)
        weights = torch.tensor(np_weights, **self.dev)
        return weights

    def _gen_uniform_weights(self, dim, connectivity = None):
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
        n, m  = dim[0], dim[1]
        weights = torch.rand(n, m, generator = self.random_state, device = self.device, requires_grad = False)
        weights =  (weights * 2) - 1
        if connectivity is not None:
            accept = torch.rand(n, m, **self.tensor_args) < connectivity
            weights *= accept
        return weights

    def _gen_reservoir(self, obs_idx = None, targ_idx = None, load_failed = None):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state

        #random_state = np.random.RandomState(self.random_state)

        max_tries = 1000  # Will usually finish on the first iteration
        n = self.n_nodes

        #if the size of the reservoir has changed, reload it.
        if self.reservoir:
            self.reservoir = ray.get(self.reservoir)
            if self.reservoir.n_nodes_ != self.n_nodes:
                load_failed = 1

        already_warned = False
        book_index = 0
        for i in range(max_tries):
            if i > 0:
                _printc(str(i), 'fail', end = ' ')

            #only initialize the reservoir and connectivity matrix if we have to for speed in esn_cv.
            if not self.reservoir or not self.approximate_reservoir or load_failed == 1:

                self.accept = torch.rand(self.n_nodes, self.n_nodes, **self.tensor_args) < self.connectivity
               
                if self.reservoir_weight_dist == "uniform":
                    self.weights = torch.rand(self.n_nodes, self.n_nodes, **self.tensor_args) * 2 - 1
                elif self.reservoir_weight_dist == "normal":
                    shape_tuple = (self.n_nodes, self.n_nodes)
                    ones_tensor, zeros_tensor = torch.ones(shape_tuple, **self.dev), torch.zeros(shape_tuple, **self.dev)
                    self.weights = torch.normal(mean = ones_tensor, std = zeros_tensor) * self.sigma + self.mu
                elif self.reservoir_weight_dist == "discrete":
                    dim = (self.n_nodes, self.n_nodes)
                    self.weights = self._gen_discrete_weights(sigma = self.reservoir_sigma, dim = dim, connectivity = self.connectivity)
                    # sigma = self.reservoir_sigma
                    # sparcity, c = (1 - self.connectivity), self.connectivity/2
                    # dim = (self.n_nodes, self.n_nodes)
                    # np_weights = np.random.choice([0, -sigma, sigma],  p = [sparcity, c, c], size = dim)
                    # self.weights = torch.tensor(np_weights, **self.dev)
                else:
                    assert False, "{self.reservoir_weight_dist} reservoir_weight_distribution not yet implimented"


                self.weights *= self.accept
                #self.weights = csc_matrix(self.weights)
            else:
                #print("LOADING MATRIX", load_failed)
                try:
                    if self.approximate_reservoir:
                        self.weights = self.reservoir.get_approx_preRes(self.connectivity, i).to(self.device)
                    else:
                        self.weights = self.reservoir.reservoir_pre_weights < self.connectivity
                        self.weights *= self.reservoir.accept
                        self.weights = self.weights

                        del self.accept; del self.reservoir.reservoir_pre_weights;

                    #_printc("reservoir successfully loaded (" + str(self.weights.shape) , 'green') 
                except:
                    assert 1 == 0
                    if not i:
                        _printc("approx reservoir " + str(i) + " failed to load ...regenerating...", 'fail')
                    #skip to the next iteration of the loop
                    if i > self.reservoir.number_of_preloaded_sparse_sets:
                        load_failed = 1
                        _printc("All preloaded reservoirs Nilpotent, generating random reservoirs, connectivity =" + str(round(self.connectivity,8)) + '...regenerating', 'fail')
                    continue
                else:
                    assert 1 == 0, "TODO, case not yet handled."

            max_eigenvalue = torch.linalg.eigvals(self.weights).abs().max() #.type(torch.float32) .sort(descending = True).values.
             
            #max_eigenvalue = self.weights.eig(eigenvectors = False)[0].abs().max()
            
            if max_eigenvalue > 0:
                break
            else: 
                if not already_warned:
                    _printc("Loaded Reservoir is Nilpotent (max_eigenvalue ={}), connectivity ={}.. .regenerating".format(max_eigenvalue, round(self.connectivity,8)), 'fail')
                already_warned = True
                #if we have run out of pre-loaded reservoirs to draw from :
                if i == max_tries - 1:
                    raise ValueError('Nilpotent reservoirs are not allowed. Increase connectivity and/or number of nodes.')

        # Set spectral radius of weight matrix
        self.weights = self.weights * self.spectral_radius / max_eigenvalue
        self.weights = Parameter(self.weights, requires_grad = False)

        self.LinRes.weight = self.weights
        
        if load_failed == 1 or not self.reservoir:
            self.state = torch.zeros(1, self.n_nodes, device=torch_device(self.device), **self.no_grad_)
        else:
            self.state = self.reservoir.state

        # Set output weights to none to indicate untrained ESN
        self.out_weights = None
             

    def _set_Win(self):
        """
        Build the input weights.
        Currently only uniform and discrete weights implimented.
        """
        with torch.no_grad():
            n, m, o = self.n_nodes, self.n_inputs, self.n_outputs
            #weight
            if not self.reservoir or 'in_weights' not in dir(self.reservoir): 
                
                #print("GENERATING IN WEIGHTS")
                if self.input_weight_dist == "uniform":
                    in_weights = self._gen_uniform_weights(dim = (n,m), connectivity = self.input_connectivity)
                    # in_weights = torch.rand(n, m, generator = self.random_state, device = self.device, requires_grad = False)
                    # in_weights =  (in_weights * 2) - 1
                    # if self.input_connectivity is not None:
                    #     accept = torch.rand(n, m, **self.tensor_args) < self.input_connectivity
                    #     in_weights *= accept
                elif self.input_weight_dist == "discrete":

                    #input_scaling is input sigma.
                    in_weights = self._gen_discrete_weights(sigma = 1, dim = (n, m), connectivity = self.input_connectivity)

                else:
                    assert False, f'input_weight_dist {self.input_weight_dist} not implimented, try uniform or discrete.'
                
            else:
                
                in_weights = self.reservoir.in_weights #+ self.noise * self.reservoir.noise_z One possibility is to add noise here, another is after activation.
                
                ##### Later for speed re-add the feedback weights here.

                if self.feedback:
                    
                    feedback_weights = self.feedback_scaling * self.reservoir.feedback_weights
                
                    #in_weights = torch.hstack((in_weights, feedback_weights)).view(self.n_nodes, -1)

            in_weights *= self.input_scaling

            #if there is white noise add it in (this will be much more useful later with the exponential model)
            #populate this bias matrix based on the noise

            #bias
            #uniform bias can be seen as means of normal random variables.
            if self.bias == "uniform":
                #random uniform distributed bias
                bias = bias * 2 - 1
            elif type(self.bias) in [type(1), type(1.5)]:
                bias = bias = torch.zeros(n, 1, device = self.device, **self.no_grad_)
                bias = bias + self.bias

                #you could also add self.noise here.
            
            self.bias_ = bias
            if self.bias_.shape[1] == 1:
                self.bias_ = self.bias_.squeeze()

            if self.feedback:

                if self.feedback_weight_dist == "uniform":
                    feedback_weights = self._gen_uniform_weights(dim = (n,o), connectivity = self.feedback_connectivity) #torch.rand(self.n_nodes, self.n_outputs, device = self.device, requires_grad = False, generator = self.random_state) * 2 - 1
                    feedback_weights *= self.feedback_scaling
                    feedback_weights = feedback_weights.view(self.n_nodes, -1)

                elif self.feedback_weight_dist == "discrete":

                    #input_scaling is input sigma.
                    feedback_weights = self._gen_discrete_weights(sigma = 1, dim = (n, o), connectivity = self.feedback_connectivity)

                else:
                    assert False, f'feedback weight dist {self.feedback_weight_dist} not implimented. Try uniform or discrete'

                
                feedback_weights = Parameter(feedback_weights, requires_grad = False)
                if self.ODE_order:
                    self.train_state = self._train_state_feedback_unsupervised
                else:
                    self.train_state = self._train_state_feedback 

            else:
                feedback_weights = None
                if self.ODE_order:
                    #self.train_state = self.train_state_feedback_unsupervised
                    self.train_state = self._train_state_vanilla_rs
                else:
                    #self.train_state = self.train_state_feedback 
                    self.train_state = self._train_state_vanilla

        if feedback_weights is not None:
            in_weights = torch.hstack((in_weights, feedback_weights))
   
        in_weights = Parameter(in_weights, requires_grad = False)
        #in_weights._name_ = "in_weights"

        return in_weights#(in_weights, feedback_weights)

    def _freeze_weights(self):
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
        names = []
        for name, param in zip(self.state_dict().keys(), self.parameters()):
            names.append(name)
            if name != "LinOut.weight":
                param.requires_grad_(False)
            else:
                self.LinOut.weight.requires_grad_(True)
                self.LinOut.bias.requires_grad_(True)
                assert self.LinOut.weight.requires_grad
            #print('param:', name,  params.requires_grad)

    def _train_states_supervised(self, X, y, states, outputs : bool = True): #calc_grads : bool = True, 
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        X : tensor
            Description
        y : tensor
            Descriptio
        states : tensor
            Description
        outputs : bool
            Description
        calc_grads: bool
            Description

        Returns
        -------
        states : tensor
            Description

        """
        #self.state_grads = []
        #self.state_list = []

        with torch.no_grad():


            for t in range(1, X.shape[0]):
                #print(t)
                input_t =  X[t, :].T
                state_t, _ = self.train_state(t, X = input_t, state = states[t-1,:], y = y[t-1,:])

                states = torch.cat([states, state_t.view(-1, self.n_nodes)], axis = 0)


        return states

    def _train_states_unsupervised(self, X, y, states): 
        """
        Internal function: Train states unsupervised.

        The unsupervised part of RcTorch solves differential equations.
        These arguments were stripped calc_grads : bool = True, outputs : bool = True

        Parameters
        ----------
        X : int
            Description of arg1
        y : str
            Description of arg2
        states : dtype
            Description
        calc_grads : bool
            Description
        outputs : bool
            Description



        Returns
        -------
        tensor
            states

        """
        #self.state_grads = []
        #self.state_list = []
        with torch.no_grad():

            #output_prev = y[0]

            for t in range(1, X.shape[0]):
                input_t =  X[t, :].T
                #assert False,f'y {y.shape} input_t {input_t.shape}'
                state_t, output_t = self.train_state(t, X = input_t,
                                                     state = states[t-1,:], 
                                                     y = y[t-1,:],
                                                     output = False)


                states = torch.cat([states, state_t.view(-1, self.n_nodes)], axis = 0)
                #output_prev = output_t

        #outputs = self.LinOut(states)
        return states

    def _extend_X(self, X, random_sampling):
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        X : dtype
            Description of arg1
        random_sampling : bool?
            Description of arg2

        Returns
        -------
        tensor
            X_extended

        """
        if self.burn_in and self.ODE_order and not random_sampling:
            start = float(X[0] - self.burn_in * self.dt)
            neg_X = torch.linspace(start, float(X[0]), self.burn_in).view(-1, 1).to(self.device)
            X_extended = torch.cat([neg_X, X], axis = 0)


        elif self.burn_in and self.ODE_order and random_sampling:

            dt_mu = torch.mean(self.dt)
            start = float(X[0] - self.burn_in * dt_mu)
            neg_X = torch.linspace(start, float(X[0]), self.burn_in).view(-1, 1).to(self.device)
            X_extended = torch.cat([neg_X, X], axis = 0)
            assert False, "implementation incomplete"

            alpha = self.leaking_rate_orig**dt_mu
            self.leaking_rate_new = (1 - alpha), alpha
        else:
            X_extended = X
        return X_extended

    def _calc_bias(self, weights):
        """
        Calculates bias

        Extended description of function.
        In particular when is this function relevant?

        Parameters
        ----------
        weights : pytorch.tensor
            weights

        Returns
        -------
        tensor
            bias tensor, shape (...)

        """
        return self._y_means - self._x_means @ weights

    def _solve_supervised(self, y, return_states, SCALE):
        """
        Solve, via ridge regularized linear regression, for the output weights.
        "supervised" means if the esn has data.

        Extended description of function.

        Parameters
        ----------
        y : torch.tensor
            target time series
        return_states : bool
            if True the function returns (states, y, burn_in)
        SCALE: bool
            whether or not to scale the data

        Returns
        -------
         if True the function returns (states, y, burn_in)
            Description of return value

        """
        train_x = self.extended_states
        if not self.random_sampling:
            n_time_points  = y.shape[0]
            bool_mask = torch.rand(n_time_points) <= self.solve_sample_prop

            
            # ones_row = torch.ones( train_x.shape[0], 1, **self.dev)
            # train_x = torch.hstack((ones_row, train_x))

            train_x = train_x[bool_mask]
            y = y[bool_mask]
            

        self._x_means = torch.mean(train_x, axis = 0)
        self._y_means = torch.mean(y, axis = 0)
        

        
        train_x = train_x - self._x_means

        biases = []
        weights = []
        
        ridge_x = torch.matmul(train_x.T, train_x) + self.regularization * torch.eye(train_x.shape[1], **self.dev)
        
        try:
            ridge_x_inv = torch.pinverse(ridge_x)
        except:
            ridge_x_inv = torch.inverse(ridge_x)


        for i in range(y.shape[1]):
            train_y = y[:, i].view(-1,1)
            train_y = train_y

   
            ridge_y = torch.matmul(train_x.T, train_y)
            weight = ridge_x_inv @ ridge_y


            #intercept line, pyds inspired
            #self.intercept_ = self._y_mean - self._x_means @ self.coef_
        
            bias = self._y_means[i] - self._x_means @ weight
            #bias = self._y_means[i] - self._x_means @ weight

            biases.append(bias)
            weights.append(weight)

        
        self.LinOut.weight = Parameter(torch.hstack(weights).T)
        self.LinOut.bias = Parameter(torch.hstack(biases).view(-1, self.n_outputs))

        # self.N = self.LinOut(self.extended_states)
                
        # # Store last y value as starting value for predictions
        # self.lastoutput = y[-1, :]

        # if self.burn_in:
        #     self.N = self.N[self.burn_in:,:]
        #     self.N = self.N.view(-1, self.n_outputs)
        #     self.X = self.X[self.burn_in:]
            
        # # Return all data for computation or visualization purposes (Note: these are normalized)
        # if return_states:
        #     return extended_states, (y[1:,:] if self.feedback else y), self.burn_in
        # else:
        #     print("burn in", self.burn_in, "ex", self.extended_states.shape, "linout", self.LinOut.weight.shape)
        #     #self.yfit = self.LinOut(self.extended_states)
        #     if SCALE:   
        #         self.yfit = self._output_stds * self.N + self._output_means
        #     if not SCALE:
        #         assert False
        #     return self.yfit
        if return_states:
            return self.extended_states, (y[1:,:] if self.feedback else y), burn_in
        else:
            yfit_norm = self.LinOut(self.extended_states) #self.LinOut.weight.cpu()@self.extended_states.T.cpu() + self.LinOut.bias.cpu()
            #yfit = self._output_stds.cpu()* (yfit_norm)+ self._output_means.cpu()

            self.yfit = self._descale(outputs = yfit_norm, normalize = self._normalize).view(-1, self.n_outputs).detach().numpy()
            return self.yfit