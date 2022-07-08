#Imports
import math
from dataclasses import dataclass

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
from torch import matmul, pinverse, hstack, eye, ones, zeros, cuda, Generator, rand, randperm, no_grad, normal, tensor, vstack, cat, dot, ones_like, zeros_like
from torch import clamp, prod, where, randint, stack
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

from .custom_loss import *
import re



#pytorch elastic net regularization:
#https://github.com/jayanthkoushik/torch-gel

#TODO: unit test setting interactive to False.

#TODO: repair esn documentation (go strait to reinier's, copy and make adjustments)

#TODO: rename some pyesn variables.

#TODO: improve documentation.
def sech2(z):
    return (1/(torch.cosh(z)))**2

class Recurrence(Function):

    @staticmethod
    def forward(ctx, states, esn, X, y, weights):
        states, states_dot = esn.train_states(X, y, states)
        ctx.states = states
        ctx.states_dot = states_dot
        return states, states_dot

    @staticmethod
    def backward(ctx, grad_output, weights):
        if grad_output is None:
            return None, None
        output = torch.matmul(ctx.states_dot, weights.T)

        return output, None, None, None, None

def dfx(x,f, retain_graph = True, create_graph = True, requires_grad = True, grad_outputs = None):
    try:
        assert not grad_outputs
        return grad([f],[x], grad_outputs=ones_like(f), 
                    create_graph = create_graph, retain_graph = retain_graph)[0]
    except:
        return grad([f],[x], grad_outputs=ones_like(f), create_graph = create_graph, 
                             retain_graph = retain_graph)[0]

def check_x(X, y, tensor_args = {}):
    if X is None:
        #X = ones(*y.shape, device = device, requires_grad = requires_grad)
        X = torch.linspace(0, 1, steps = y.shape[0], **tensor_args)
    try:
        if X is None:
            X = ones(*y.shape, **tensor_args)
            #print("1a")
    except:
        print("first try-catch failed")
        pass
    try:
        if type(X) == np.ndarray:
            X = torch.tensor(X,  **tensor_args)
            #print("2a")
    except:
        pass
    try:
        if len(X.shape) == 1:
            X = X.view(-1, 1)
            #print("3a")
    except:
        pass
    #assert X.requires_grad == tensor_args["requires_grad"]
    return X


def printn(param: torch.nn.parameter):
    """TODO"""
    print(param._name_ + "\t \t", param.shape)

def NRMSELoss(yhat,y):
    """TODO"""
    return torch.sqrt(torch.mean((yhat-y)**2)/y**2)

def sinsq(x):
    """TODO"""
    return torch.square(torch.sin(x))

def printc(string_, color_, end = '\n') :
    """TODO"""
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
    print(colorz[color_] + string_ + colorz["endc"] , end = end)

def convert_ode_coefs(ode_coefs_, X):
    #print('type_X', type_X)
    ode_coefs = ode_coefs_.copy()
    if type(ode_coefs_) == list:
        for i, coef in enumerate(ode_coefs_):
            if type(coef) == str:
                if coef[0] == "t" and (coef[1] == "^" or (coef[1] == "*" and coef[2] == "*")):
                    pow_ = float(re.sub("[^0-9.-]+", "", coef))
                    ode_coefs[i]  = X ** pow_
            elif type(coef) in [float, int, type(X)]:
                pass
            else:
                assert False, "ode_coefs must be a list of floats or strings of the form 't^pow', where pow is a real number."
    else:
        assert False, "ode_coefs must be a list of floats or strings of the form 't^pow', where pow is a real number."
    return ode_coefs

tanh_activation = Tanh()
class EchoStateNetwork(nn.Module):
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
    
    
    BACKPROP ARGUMENTS (not needed for the homework)
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



    Methods
    -------
    train(y, x=None, burn_in=100)
        Train an Echo State Network
    test(y, x=None, y_start=None, scoring_method='mse', alpha=1.)
        Tests and scores against known output
    predict(n_steps, x=None, y_start=None)
        Predicts n values in advance
    predict_stepwise(y, x=None, steps_ahead=1, y_start=None)
        Predicts a specified number of steps into the future for every time point in y-values array (NOT IMPLIMENTED)

    Arguments to be implimented later:
        obs_idx = None, resp_idx = None, input_weight_type = None, model_type = "uniform", PyESNnoise=0.001, 
        regularization lr: reg_lr = 10**-4, 
        change bias back to "uniform"
    """
    def __init__(self, n_nodes = 1000, bias = 0, connectivity = 0.1, leaking_rate = 0.99, spectral_radius=0.9, #<-- important hyper-parameters
                 regularization = None, feedback = False,                              #<-- activation, feedback
                 input_scaling = 0.5, feedback_scaling = 0.5, noise = 0.0,                                     #<-- hyper-params not needed for the hw
                 approximate_reservoir = False, device = None, id_ = None, random_state = 123, reservoir = None, #<-- process args
                 classification = False, l2_prop = 1, n_inputs = 1, n_outputs = 1,
                  dtype = torch.float32, calculate_state_grads = False, dt = None,
                 activation_f = tanh_activation, act_f_prime = sech2,  enet_strength = None,
                 gamma = None, spikethreshold = None,
                 enet_alpha = None, gamma_cyclic = None): #<-- this line is backprop arguments
        super().__init__()
        
        #activation function
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.activation_function = activation_f
        self.act_f_prime = act_f_prime
        self.enet_strength = enet_strength
        self.enet_alpha = enet_alpha

        if self.enet_alpha:
            assert self.enet_strength > 0
            assert self.enet_alpha >= 0 and self.enet_alpha <=1

        if activation_f != tanh_activation and act_f_prime == sech2:
            assert False, f'your activation f is not tanh but act_f_prime is sech2'


        self.calculate_state_grads = calculate_state_grads
        self.dt = dt

        #cuda (gpu)
        if not device:
            self.device = torch_device("cuda" if cuda_is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype

        # random state and default tensor arguments
        self.random_state = Generator(device=self.device).manual_seed(random_state)
        self.no_grad_ = {"requires_grad" : False}
        self.tensor_args = {"device": self.device, "generator" : self.random_state, **self.no_grad_}

        # hyper-parameters:
        self.bias = bias
        self.connectivity = connectivity
        self.feedback_scaling = feedback_scaling
        self.input_scaling = input_scaling
        self.leaking_rate = [leaking_rate, 1 - leaking_rate]
        self.n_nodes = n_nodes
        self.noise = noise
        self.regularization = regularization
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.gamma_cyclic = gamma_cyclic
        self.spikethreshold = spikethreshold

        #Feedback
        self.feedback = feedback

        #For speed up: approximate implimentation and preloaded reservoir matrices.
        self.approximate_reservoir, self.reservoir = approximate_reservoir, reservoir

        #elastic net attributes: (default is 1, which is ridge regression for speed)
        self.l2_prop = l2_prop

        self.id_ = id_
        
        #Reservoir layer
        self.LinRes = Linear(self.n_nodes, self.n_nodes, bias = False)

        #https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        self.classification = classification

        self.LinIn = Linear(self.n_inputs, self.n_nodes,  bias = False)
        self.LinFeedback = Linear(self.n_inputs, self.n_nodes, bias = False)
        self.LinIn.weight, self.LinFeedback.weight = self.set_Win()
        torch.manual_seed(random_state)
        self.LinOut = Linear(self.n_nodes + 1, self.n_outputs)

        """
        if self.classification:
            self.log_reg = Linear(self.n_nodes, 2)
            #self.criterion = criterion #torch.nn.CrossEntropyLoss()
        else:
            #self.criterion = MSELoss()
        """
            
        with no_grad():
            self.gen_reservoir()

        self.dev = {"device" : self.device, "dtype" : self.dtype, "requires_grad" : False}#, "requires_grad": self.track_in_grad}
        
        #scaler = "standardize"
        #if scaler == "standardize":
        #    self.scale   = self.stardardize
        #    self.descale = self.destandardize

        """TODO: additional hyper-parameters
        noise from pyesn — unlike my implimentation it happens outside the activation function. 
        TBD if this actually can improve the RC.
        self.PyESNnoise = 0.001
        self.external_noise = rand(self.n_nodes, device = self.device)
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
        }"""

    def plot_reservoir(self):
        """Plot the network weights"""
        sns.histplot(self.weights.cpu().numpy().view(-1,))

    def train_state(self, t, X, state, y, output = False, retain_grad = False):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.
        """
        # generator = self.random_state, device = self.device) 

        input_vector = self.LinIn(X)
        recurrent_vec = self.LinRes(state)

        preactivation = input_vector + recurrent_vec + self.bias

        # if self.feedback:
        #     assert X.shape == y.shape, f'{X.shape} != {y.shape}'
        #     #avoiding inplace operations:
        #     feedback_vec = self.LinFeedback(y)
        #     assert feedback_vec.shape == preactivation.shape, f'{feedback_vec.shape} != {preactivation.shape}'
        #     preactivation = preactivation.clone() + feedback_vec


        update = self.activation_function(preactivation)
        next_state = self.leaking_rate[0] * update + self.leaking_rate[1] * state
        # if output:
        #     return next_state, self.LinOut(cat([X, next_state], axis = 0).view(self.n_outputs,-1))
        # else:
        return next_state, None

    def forward(self, extended_states):
        """
        if self.burn_in:
            #extended_states = extended_states[self.burn_in:]
            extended_states = torch.cat((self.extended_states[0,:].view(1,-1), self.extended_states[(self.burn_in + 1):,:]), axis = 0)
        """
        output = self.LinOut(extended_states)
        return output

    def calc_Ndot(self, states_dot, cutoff = True):
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



    def gen_reservoir(self, obs_idx = None, targ_idx = None, load_failed = None):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state

        #random_state = np.random.RandomState(self.random_state)

        max_tries = 1000  # Will usually finish on the first iteration
        n = self.n_nodes

        #if the size of the reservoir has changed, reload it.
        if self.reservoir:
            if self.reservoir.n_nodes_ != self.n_nodes:
                load_failed = 1

        already_warned = False
        book_index = 0
        for i in range(max_tries):
            if i > 0:
                printc(str(i), 'fail', end = ' ')

            #only initialize the reservoir and connectivity matrix if we have to for speed in esn_cv.
            if not self.reservoir or not self.approximate_reservoir or load_failed == 1:

                self.accept = rand(self.n_nodes, self.n_nodes, **self.tensor_args) < self.connectivity
                self.weights = rand(self.n_nodes, self.n_nodes, **self.tensor_args) * 2 - 1
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

                    #printc("reservoir successfully loaded (" + str(self.weights.shape) , 'green') 
                except:
                    assert 1 == 0
                    if not i:
                        printc("approx reservoir " + str(i) + " failed to load ...regenerating...", 'fail')
                    #skip to the next iteration of the loop
                    if i > self.reservoir.number_of_preloaded_sparse_sets:
                        load_failed = 1
                        printc("All preloaded reservoirs Nilpotent, generating random reservoirs, connectivity =" + str(round(self.connectivity,8)) + '...regenerating', 'fail')
                    continue
                else:
                    assert 1 == 0, "TODO, case not yet handled."
             
            max_eigenvalue = self.weights.eig(eigenvectors = False)[0].abs().max()
            
            if max_eigenvalue > 0:
                break
            else: 
                if not already_warned:
                    printc("Loaded Reservoir is Nilpotent (max_eigenvalue ={}), connectivity ={}.. .regenerating".format(max_eigenvalue, round(self.connectivity,8)), 'fail')
                already_warned = True
                #if we have run out of pre-loaded reservoirs to draw from :
                if i == max_tries - 1:
                    raise ValueError('Nilpotent reservoirs are not allowed. Increase connectivity and/or number of nodes.')

        # Set spectral radius of weight matrix
        self.weights = self.weights * self.spectral_radius / max_eigenvalue
        self.weights = Parameter(self.weights, requires_grad = False)

        self.LinRes.weight = self.weights
        
        if load_failed == 1 or not self.reservoir:
            self.state = zeros(1, self.n_nodes, device=torch_device(self.device), **self.no_grad_)
        else:
            self.state = self.reservoir.state

        # Set output weights to none to indicate untrained ESN
        self.out_weights = None
             

    def set_Win(self): #inputs
        """
        Build the input weights.
        Currently only uniform implimented.

        Arguments:
            inputs:
        """
        with no_grad():
            n, m = self.n_nodes, self.n_inputs
            #weight
            if not self.reservoir or 'in_weights' not in dir(self.reservoir): 
                
                #print("GENERATING IN WEIGHTS")

                in_weights = rand(n, m, generator = self.random_state, device = self.device, requires_grad = False)
                in_weights =  in_weights * 2 - 1
                
            else:
                
                in_weights = self.reservoir.in_weights #+ self.noise * self.reservoir.noise_z One possibility is to add noise here, another is after activation.
                
                ##### Later for speed re-add the feedback weights here.

                #if self.feedback:
                #    feedback_weights = self.feedback_scaling * self.reservoir.feedback_weights
                #    in_weights = hstack((in_weights, feedback_weights)).view(self.n_nodes, -1)

            in_weights *= self.input_scaling

            #if there is white noise add it in (this will be much more useful later with the exponential model)
            #populate this bias matrix based on the noise

            #bias
            #uniform bias can be seen as means of normal random variables.
            if self.bias == "uniform":
                #random uniform distributed bias
                bias = bias * 2 - 1
            elif type(self.bias) in [type(1), type(1.5)]:
                bias = bias = zeros(n, 1, device = self.device, **self.no_grad_)
                bias = bias + self.bias

                #you could also add self.noise here.
            
            self.bias_ = bias
            if self.bias_.shape[1] == 1:
                self.bias_ = self.bias_.squeeze()

            if self.feedback:
                feedback_weights = rand(self.n_nodes, self.n_outputs, device = self.device, requires_grad = False, generator = self.random_state) * 2 - 1
                feedback_weights *= self.feedback_scaling
                feedback_weights = feedback_weights.view(self.n_nodes, -1)
                feedback_weights = Parameter(feedback_weights, requires_grad = False) 
            else:
                feedback_weights = None
   
        in_weights = Parameter(in_weights, requires_grad = False)
        #in_weights._name_ = "in_weights"

        return (in_weights, feedback_weights)
    
    def check_device_cpu(self):
        """TODO: make a function that checks if a function is on the cpu and moves it there if not"""
        pass

    def display_in_weights(self):
        """TODO"""
        sns.heatmap(self.in_weights)

    def display_out_weights(self):
        """TODO"""
        sns.heatmap(self.out_weights)

    def display_res_weights(self):
        """TODO"""
        sns.heatmap(self.weights)

    def plot_states(self, n= 10):
        """TODO"""
        for i in range(n):
            plt.plot(list(range(len(self.state[:,i]))), RC.state[:,i], alpha = 0.8)

    def freeze_weights(self):
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

    def train_states(self, X, y, states, calc_grads = True, outputs = True):
        #self.state_grads = []
        #self.state_list = []
        with no_grad():
            inputs = []
            if self.ODE_order:
                if not self.alpha:
                    self.alpha = self.leaking_rate[0] / self.dt

            for t in range(0, X.shape[0]):
                input_t =  X[t, :].T
                state_t, _ = self.train_state(t, X = input_t,
                                          state = states[t,:], 
                                          y = None, #y[t,:],
                                          output = False)

                states = cat([states, state_t.view(-1, self.n_nodes)], axis = 0)

        return states

    def extend_X(self, X):
        if self.burn_in and self.ODE_order:
            start = float(X[0] - self.burn_in * self.dt)
            neg_X = torch.linspace(start, float(X[0]), self.burn_in).view(-1, 1).to(self.device)
            X_extended = torch.cat([neg_X, X], axis = 0)
        else:
            X_extended = X
        return X_extended


    def fit(self, y = None, X=None, burn_in=300, input_weight=None, verbose = False , 
            learning_rate = 0.005, return_states = False, criterion =  MSELoss(), 
            optimizer = None, out_weights = None, ODE_order = None, SOLVE = True,
            reparam_f = None, init_conditions = None, force = None, #nl_f = None, 
            ode_coefs = None, train_score = False, ODE_criterion = None, 
            preloaded_states_dict = None, q = None, eq_system = None, nl = False,
            backprop_f = None, epochs = None):
        # if verbose:
        #     if nl:
        #         print("nonlinear!")
        #     else:
        #         print("linear")


        """
        NLLLoss(),
        Train the network.
        
        Arguments: TODO
            y: response matrix
            x: observer matrix
            burn in: obvious
            input_weight : ???
            learning_rate: 
            verbose:
        """

        with no_grad():
            self.reparam = reparam_f
            self.q = q
            self.burn_in = burn_in
            self.criterion = criterion
            self.track_in_grad = False #track_in_grad

            self.ode_coefs = ode_coefs

            #assert len(init_conditions) == ODE_order
            self.ODE_order = ODE_order

            if not self.ODE_order:
                SCALE = True
            else:
                SCALE = False



            #if save_after_n_epochs  == None and self.epochs:
            #    save_after_n_epochs = int(self.epochs * 0.7)
            

            #if self.id_ is None:
            #    assert y.is_leaf == True, "y must be a leaf variable"
            
            #ensure that y is a 2-dimensional tensor with pre-specified arguments.
            if self.ODE_order: 
                assert init_conditions
                if type(init_conditions[0]) == list or type(init_conditions[0]) == np.ndarray:
                    multiple_ICs = self.multiple_ICs = True
                elif type(init_conditions[0]) in [float,  int, np.float64, np.float32, np.int32, np.float64]:
                    multiple_ICs = self.multiple_ICs = False
                else:
                    assert False, f'malformed ICs, either use a list, a float or an integer'
                init_conds = init_conditions
                
                self.eq_system = eq_system

                if self.dt != None:
                    self.alpha = self.leaking_rate[0] / self.dt

                    start, stop = float(X[0]), float(X[-1])
                    nsteps = int((stop - start) / self.dt)
                    X = torch.linspace(start, stop, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
                elif type(X) == type([]) and len(X) == 3:
                    x0, xf, nsteps = X #6*np.pi, 100
                    X = torch.linspace(x0, xf, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
                else:
                    assert False, "Please input start, stop, dt"
                y= ones_like(X)

                self.init_conds = init_conditions.copy()
                # for i, condition in enumerate(self.init_conditions):
                #     if not multiple_ICs:

                #         if type(condition) == float or type(condition) == int:
                #             init_conds.append(self.init_conditions[i])
                #         else:
                            
                #             #assert 1==2, condition
                #             rand_init_cond = float(torch.FloatTensor(1, 1).uniform_(condition[0], condition[1]))
                #             init_conds.append(rand_init_cond)

                #     else:
                #         init_conds = self.init_conditions
                ode_coefs = convert_ode_coefs(self.ode_coefs, X)
                
            else:
                #ensure that X is a two dimensional tensor, or if X is None declare a tensor.
                X = check_x(X, y, self.dev).to(self.device)
                X.requires_grad_(False)
                if type(y) == np.ndarray:
                     y = torch.tensor(y, **self.dev)
                y = y.clone()
                if y.device != self.device:
                    y = y.to(self.device)
                if len(y.shape) == 1:
                    y = y.view(-1, 1)
                if SCALE:
                    y = self.scale(outputs=y, keep=True)    
                y.requires_grad_(False)
                self.lastoutput = y[-1, :]

            self.unscaled_X = Parameter(X, requires_grad = self.track_in_grad)

            if self.unscaled_X.device != self.device:
                self.unscaled_X.data = self.unscaled_X.data.to(self.device)

            #assert self.unscaled_X.requires_grad == self.track_in_grad
            
            #protect against input that is a vector of ones. (otherwise we divide by infinity)
            # Normalize inputs and outputs
            

            if self.unscaled_X.std() != 0:
                self.X = self.unscaled_X.clone()

                if SCALE:
                    self.X.data = self.scale(inputs = self.unscaled_X, keep = True).clone()
            else:
                self._input_stds = None
                self._input_means = None

            self.X_extended = self.extend_X(self.X)

            start_index = 1 if self.feedback else 0 
            rows = X.shape[0] - start_index
            
            self.n_inputs = self.X.shape[1] 
            
            self.lastinput = self.X[-1, :]
            
            start_index = 1 if self.feedback else 0 
            rows = X.shape[0] - start_index

            #self.LinOutDE = Linear(self.n_nodes + 1, self.n_outputs)

            

            ################################################################################################
            #+++++++++++++++++++++++++++         FORWARD PASS AND SOLVE             ++++++++++++++++++++++++
            ################################################################################################
            with torch.set_grad_enabled(self.track_in_grad):
                self.freeze_weights()
                if out_weights:
                #assert  False
                    self.SOLVE = SOLVE = False
                    self.LinOut.weight =  Parameter(out_weights["weights"].to(self.device))
                    self.LinOut.bias = Parameter(out_weights["bias"].to(self.device)) 
                else:
                    try:
                        assert self.LinOut.weight.device == self.device
                    except:
                        self.LinOut.weight = Parameter(self.LinOut.weight.to(self.device))
                        self.LinOut.bias = Parameter(self.LinOut.bias.to(self.device))

                if not preloaded_states_dict:

                    self.states = zeros((1, self.n_nodes), **self.dev)
                    
                    # CUSTOM_AUTOGRAD_F = False
                    # if CUSTOM_AUTOGRAD_F:
                    #     #for more information on custom pytorch autograd functions see the following link:
                    #     #https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
                    #     recurrence = Recurrence.apply

                    #     self.states, self.states_dot, states_dot = recurrence(self.states, self, self.X_extended, y)
                    # else:
                    self.states = self.train_states(self.X_extended, y, self.states)

                    #drop the first state and burned states
                    self.states = self.states[1:]
                    if self.burn_in:
                        self.states = self.states[self.burn_in:]

                    updates = self.LinIn(self.X) + self.bias + self.LinRes(self.states)

                    # calculate hidden state derivatives
                    if self.ODE_order:
                        self.states_dot = - self.alpha * self.states + self.alpha * self.activation_function(updates)
                        # if self.ODE_order == 2:
                        #     self.states_dot2 = - self.alpha * self.states_dot + self.alpha * self.act_f_prime(updates) * (self.LinIn.weight.T + self.bias + self.LinRes(self.states_dot))
                        #     self.states_dot2 = torch.cat((zeros_like(self.X), self.states_dot2), axis = 1)

                        self.states_dot = torch.cat((ones_like(self.X), self.states_dot), axis = 1)

                    self.G = G = self.reparam(self.X)#, order = self.ODE_order)

                    #append columns for the data:
                    self.extended_states = torch.cat((self.X, self.states), axis = 1)

                    del self.states

                    #add rows corresponding to bias to the states 
                    self.sb = states_with_bias = torch.cat((ones_like(self.extended_states[:,0].view(-1,1)), self.extended_states), axis = 1)
                    self.sb1 = states_dot_with_bias = torch.cat((zeros_like(self.states_dot[:,0].view(-1,1)), self.states_dot), axis = 1)

                    # do the same for the second derivatives
                    # if self.ODE_order == 2:
                    #     self.sb2 = states_dot2_with_bias = torch.cat((zeros_like(self.states_dot2[:,0].view(-1,1)), self.states_dot2), axis = 1)
                    g, g_dot = G
                    self.g = g
                else:
                    sd = preloaded_states_dict
                    self.states, self.states_dot, G, self.extended_states = sd["s"], sd["s1"], sd["G"], sd["ex"]
                    states_with_bias, states_dot_with_bias = sd["sb"], sd["sb1"]
                    # if self.ODE_order == 2:
                    #     self.states_dot2 = sd["s2"]
                    #     states_dot2_with_bias = sd["sb2"]
                    g, g_dot = G
                    self.g = g

                self.laststate = self.extended_states[-1, 1:]
                self.force = force
                self.force_t = force(self.X)

                
                with no_grad():
                    if SOLVE and not out_weights:
                        #include everything after burn_in 
                        if not self.ODE_order:
                            train_x = self.extended_states
                            train_y = y

                            if not self.regularization:
                                print("no regularization")
                                pinv = pinverse(train_x)
                                weight = matmul(pinv, train_y)

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
                                    for i in range(self.ODE_order):                                
                                        A_i = init_conds[i] * g.pow(i)# * ode_coefs[i]
                                        if not i:
                                            self.A = A = [A_i] # + ode_coefs[1] * v0 * g.pow(1) + ... + ode_coefs[m] * accel_0 * g.pow(m)
                                        else:
                                            A.append(A_i)

                                t = self.X
                                H, H_dot = states_with_bias, states_dot_with_bias
                                self.gH = gH = g * H
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

                                    self.Sigma=Sigma = sigma1 + self.regularization * eye(sigma1.shape[1], **self.dev)

                                    self.Delta=Delta = S.T @ S_dot - S_dot.T @ S

                                    #try:
                                    delta_inv = pinverse(Delta)
                                    sigma_inv = pinverse(Sigma)
                                    D_H = Sigma @ delta_inv + Delta @ sigma_inv
                                    self.Lam=Lam = pinverse(D_H)
                                    
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
                                            Wp =  Lam.T @ (Cp @ delta_inv  - Cx @  sigma_inv).T/nX
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
                                            #########################################################
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
                                        


                                    
                                    #if self.ODE_order == 1:
                                        # 

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
                                    if multiple_ICs and nl:
                                        DH1s = [DH.T @ DH for DH in DHs]

                                        #gH_sq = gH.T @ gH 
                                        #DH1p2s = [-2 * self.q* D_A.T * gH_sq for D_A in D_As]
                                        nl_corrections = [2 * self.q* D_A.T * gH.T @ gH  for D_A in D_As]
                                        #self.DH1 -2 * self.q* D_A.T * gH.T @ gH

                                        self.DH1s = DH1s = [DH1s[i] - 0 for i, correction in enumerate(nl_corrections)]
                                        
                                        #DH1 = DH1 + self.regularization * eye(DH1.shape[1], **self.dev)
                                        #reg = 
                                        DHinvs = [pinverse(DH1 + self.regularization * eye(DH1s[0].shape[1], **self.dev))for DH1 in DH1s]
                                        self.DH2s = DH2s = [DHinv @ DHs[i].T for i, DHinv in enumerate(DHinvs)]

                                        weights, biases = [], []

                                        for i in range(len(D_As)):
                                            DH2 = DH2s[i]
                                            D_A = D_As[i]
                                            init_weight = matmul(-DH2, D_A)

                                            biases.append(init_weight[0])
                                            weights.append(init_weight[1:])
                                        #assert False
                                        self.biases_list = biases
                                        self.weights_list = weights

                                    else:
                                        DH1 = DH.T @ DH

                                        #if nl:
                                        #    DH1 = DH1 -2 * self.q* D_A.T * gH.T @ gH
                                        
                                        self.DH1 = DH1 = DH1 + self.regularization * eye(DH1.shape[1], **self.dev)
                                        DHinv = pinverse(DH1)
                                        self.DH2 = DH2 = DHinv @ DH.T
                                        if not multiple_ICs:
                                            weight = matmul(-DH2, D_A)

                                            bias = weight[0]
                                            weight = weight[1:]
                                        else:
                                            weights, biases = [], []

                                            for D_A in D_As:
                                                weight = matmul(-DH2, D_A)
                                                biases.append(weight[0])
                                                weights.append(weight[1:])
                                            self.biases_list = biases
                                            self.weights_list = weights


                            elif not self.ODE_order:
                                ones_row = ones( train_x.shape[0], 1, **self.dev)
                            
                                ridge_x = matmul(train_x.T, train_x) + \
                                                   self.regularization * eye(train_x.shape[1], **self.dev)

                                ridge_y = matmul(train_x.T, train_y)

                                ridge_x_inv = pinverse(ridge_x)
                                weight = ridge_x_inv @ ridge_y

                                bias = weight[0]
                                weight = weight[1:]
                        if not multiple_ICs:
                            self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                            self.LinOut.bias = Parameter(bias.view(1, self.n_outputs))

            if multiple_ICs:
                if not backprop_f:
                    init_conds_clone = init_conditions.copy()
                    ys = []
                    ydots = []
                    scores = []
                    #init_conds_clone = init_conds.copy()
                    for i, weight in enumerate(self.weights_list):
                        #print("w", i)
                        self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                        self.LinOut.bias = Parameter(biases[i].view(1, self.n_outputs))
                        self.init_conds[0] = float(init_conds_clone[0][i])


                        N = self.LinOut(self.extended_states)
                        N_dot = self.calc_Ndot(self.states_dot, cutoff = False)
                        yfit = g*N


                        """
                        self.yfit = g*self.N
                        self.lastoutput = self.yfit[-1, :]
                        
                        for i, cond in enumerate(self.init_conds):
                            self.yfit[:, i] = self.yfit[:,i] + cond

                        self.ydot = g_dot * self.N +  g * N_dot
                        
                        """
                        
                        if not eq_system:
                            yfit[:, 0] = yfit[:,0] + init_conds[0][i]
                        else:
                            yfit = init_conds[0] + g.pow(1) * N
                            self.lastoutput = yfit[-1, :]
                            ydot = g_dot * N +  g * N_dot
                            for i, cond in enumerate(init_conds):
                                yfit[:, i] = yfit[:,i] + cond
                        #yfit[:, 1] = yfit[:,1] + init_conds[1]#[i]
                        ys.append(yfit)

                        if train_score:
                            ydot = g_dot * N +  g * N_dot
                            ydots.append(ydot)
                            score = ODE_criterion(X, yfit.data, ydot.data, self.LinOut.weight.data, 
                                                    ode_coefs = ode_coefs, init_conds = init_cond_list, 
                                                    enet_strength = self.enet_strength, enet_alpha = self.enet_alpha,
                                                    force_t = self.force_t)
                            scores.append(score)

                    self.init_conds = init_conditions
                    self.lastoutput = y[-1, :]
                    if train_score:
                        return {"scores" : scores, 
                                "weights": weights, 
                                "biases" : biases,
                                "ys"     : ys,
                                "ydots"  : ydots}
                    else:
                        return ys, ydots
                else:

                    gd_weights = []
                    gd_biases = []
                    ys = []
                    ydots =[]
                    scores = []
                    Ls = []
                    init_conds_clone = init_conditions.copy()
                    if not SOLVE:
                        orig_weights = self.LinOut.weight.clone()
                        orig_bias = self.LinOut.bias.clone()
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
                        with torch.enable_grad():
                            weight_dict = backprop_f(self, epochs = epochs)

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
                    self.init_conds = init_conditions
                    self.lastoutput = y[-1, :]
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
                self.N = self.LinOut(self.extended_states)
                
                # Store last y value as starting value for predictions
                self.lastoutput = y[-1, :]

                if self.ODE_order >= 1:
                    #calc Ndot just uses the weights
                    #self.states_dot @ self.LinOut.weight
                    
                    N_dot = self.calc_Ndot(self.states_dot, cutoff = False)
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
            

            if not ODE_order and burn_in:
                self.N = self.N[self.burn_in:,:]
                self.N = self.N.view(-1, self.n_outputs)
                self.X = self.X[self.burn_in:]
            
            # Return all data for computation or visualization purposes (Note: these are normalized)
            if return_states:
                return extended_states, (y[1:,:] if self.feedback else y), burn_in
            else:
                self.yfit = self.LinOut(self.extended_states)
                if SCALE:   
                    self.yfit = self._output_stds * self.yfit + self._output_means
                return self.yfit


    # def calculate_n_grads(self, X, y,  n = 2, scale = False):
    #     self.grads = []

    #     #X = X.reshape(-1, self.n_inputs)

    #     assert y.requires_grad, "entended doesn't require grad, but you want to track_in_grad"
    #     for i in range(n):
    #         print('calculating derivative', i+1)
    #         if not i:
    #             grad = dfx(X, y)
    #         else:
    #             grad = dfx(X, self.grads[i-1])

    #         self.grads.append(grad)

    #         if scale:
    #             self.grads[i] = self.grads[i]/(self._input_stds)
    #     with no_grad():
    #         self.grads = [self.grads[i][self.burn_in:] for i in range(n)]
                
    #         #self.yfit = self.yfit[self.burn_in:]
    #     #assert extended_states.requires_grad, "entended doesn't require grad, but you want to track_in_grad"
    
    def scale(self, inputs=None, outputs=None, keep=False):
        """Normalizes array by column (along rows) and stores mean and standard devation.

        Set `store` to True if you want to retain means and stds for denormalization later.

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
        # Checks
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')

        # Storage for transformed variables
        transformed = []
        if not inputs is None:

            if keep:
                # Store for denormalization
                self._input_means = inputs.mean(axis=0)
                self._input_stds = inputs.std(dim = 0)

            # Transform
            if not self.ODE_order:
                transformed.append((inputs - self._input_means) / self._input_stds)
            else: 
                transformed.append( inputs / self._input_stds)

        if not outputs is None:
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
        #if not torch.all(torch.isfinite(predicted)):
        #    # print("Warning: some predicted values are not finite")
        #    errors = torch.inf
        
        def nmse(y, yhat):
            """
            normalized mean square error
            """
            return ((torch.sum(torch.square(y - yhat)) / torch.sum(torch.square(y)))) / len(y.squeeze())
            
        #### attempt at loss function when steps ahead > 2 

        def step_ahead_loss(y, yhat, plot = False, decay = 0.9):
            loss = zeros(1,1, device = self.device)
            losses = []
            total_length = len(y)
            for i in range(1, total_length - self.steps_ahead):
                #step ahead == i subsequences
                #columnwise
                #   yhat_sub = yhat[:(total_length - i), i - 1]
                #   y_sub = y[i:(total_length),0]
                #row-wise
                yhat_sub = yhat[i-1, :]
                y_sub = y[i:(self.steps_ahead + i),0]
                assert(len(yhat_sub) == len(y_sub)), "yhat: {}, y: {}".format(yhat_sub.shape, y_sub.shape)

                loss_ = nmse(y_sub.squeeze(), yhat_sub.squeeze())

                if decay:
                    loss_ *= (decay ** i)

                #if i > self.burn_in:
                loss += loss_
                losses.append(loss_)

            if plot:
                plt.plot(range(1, len(losses) + 1), losses)
                plt.title("loss vs step ahead")
                plt.xlabel("steps ahead")
                plt.ylabel("avg loss")
            return loss.squeeze()

        if predicted.shape[1] != 1:
            return step_ahead_loss(y = target, yhat = predicted) 

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
        return error.type(self.dtype)
    

    # def back(self, tensor_spec, retain_graph = True):
    #     return tensor_spec.backward(torch.ones(*tensor_spec.shape, device = tensor_spec.device), retain_graph = retain_graph)

    def test(self, y, X=None, y_start=None, steps_ahead=None, 
                  scoring_method='nmse', alpha=1., scale = False, criterion = None, reparam = None,
                  ODE_criterion = None):
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
        self.reparam = reparam

        self.steps_ahead = steps_ahead

        if not self.ODE_order:
            assert False, "SCALING"

            scale = True
            if type(y) == np.ndarray:
                y = torch.tensor(y, **self.dev)
            if len(y.shape) == 1:
                y = y.view(-1, 1)
            if y.device != self.device:
                y = y.to(self.device)
            final_t =y.shape[0]
            X = check_x(X , y, self.dev).detach().clone().to(self.device).requires_grad_(True)
        else:
            if self.dt != None:
                self.alpha = self.leaking_rate[0] / self.dt

                start, stop = float(X[0]), float(X[-1])
                nsteps = int((stop - start) / self.dt)
                X = torch.linspace(start, stop, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
            elif type(X) == type([]) and len(X) == 3:
                x0, xf, nsteps = X #6*np.pi, 100
                X = torch.linspace(x0, xf, steps = nsteps, requires_grad=False).view(-1, 1).to(self.device)
            final_t =X.shape[0]

            ode_coefs = convert_ode_coefs(self.ode_coefs, X)

        X.requires_grad_(False)

        assert X.requires_grad == False#self.track_in_grad
        
        # Run prediction

        val_force_t = self.force(X)
        
        if steps_ahead is None:
            if not self.ODE_order:
                y_predicted = self.predict(n_steps = y.shape[0], X=X, y_start=y_start, scale = scale)
            else:
                assert not scale
                if not self.multiple_ICs:
                    if self.ODE_order == 1:
                        y_predicted, ydot = self.predict(n_steps = X.shape[0], X=X, y_start=y_start, scale = scale,
                                   continue_force = True)
                    elif self.ODE_order == 2:
                        y_predicted, ydot, ydot2 = self.predict(n_steps = X.shape[0], X=X, y_start=y_start, scale = scale,
                                   continue_force = True)
                    if self.ODE_order == 1:
                        score = ODE_criterion(X, y_predicted.data, ydot.data, 
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
                    if self.id_ == None:
                        #user friendly
                        if self.ODE_order:
                            return score, {"yhat": N.data, "ytest": N_dot.data}, x[self.burn_in:]
                        else:
                            return score, {"yhat": y_predicted.data, "ytest": y}, x[self.burn_in:]
                    else:
                        return score.detach(), y_predicted.detach(), self.id_

                else:
                    y_preds, ydots = self.predict(n_steps = X.shape[0], X=X, y_start=y_start, scale = scale,
                                   continue_force = True)
                    scores = []
                    for i, pred in enumerate(y_preds):
                        ydot = ydots[i]
                        if not self.eq_system:
                            score = ODE_criterion(X, pred.data, ydot.data, 
                                                  self.LinOut.weight.data, 
                                                  ode_coefs = ode_coefs, 
                                                  init_conds = self.init_conds,
                                                  enet_alpha = self.enet_alpha, 
                                                  enet_strength = self.enet_strength,
                                                  force_t = val_force_t)
                        else:
                            init_conds_system = [self.init_conds[0][i], self.init_conds[1]]
                            score = ODE_criterion(X, pred.data, ydot.data, 
                                                  self.LinOut.weight.data, 
                                                  ode_coefs = ode_coefs, 
                                                  init_conds = init_conds_system,
                                                  enet_alpha = self.enet_alpha, 
                                                  enet_strength = self.enet_strength,
                                                  force_t = val_force_t)
                        scores.append(score)
                    
                    return scores, pred.detach(), self.id_



                #y_predicted, ydot = self.reparam(self.y0, X, N, N_dot)
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            assert False, f'predict_stepwise not implimented'
            y_predicted = self.predict_stepwise(y, X, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]

    def predict(self, n_steps, X=None, y_start=None, continuation = True, scale = False, continue_force = True):
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

        

        # Normalize the inputs (like was done in train)
        if not X is None:
            if scale:
                self.X_val = Parameter(self.scale(inputs=X))
            else:
                self.X_val = Parameter(X)

        self.X_val_extended = self.X_val#self.extend_X(self.X_val)
        # if not continue_force:
        #     if self.ODE_order:
        #         continuation = False
        dev = {"device" : self.device, "dtype" : self.dtype, "requires_grad": False}

        n_samples = self.X_val_extended.shape[0]

        if not y_start is None: #if not x is None:
            if scale:
                previous_y = self.scale(outputs=y_start)[0]
            else:
                previous_y = y_start[0]

        if continuation:
            #if self.ODE_order >=2:
            #    lasthdot2 = self.lasthdot2
            #lasthdot = self.lasthdot
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            #if self.ODE_order >=2:
            #    lasthdot2 = zeros(self.n_nodes, **dev)
            #lasthdot = zeros(self.n_nodes, **dev)
            laststate = zeros(self.n_nodes, **dev)
            lastinput = zeros(self.n_inputs, **dev)
            lastoutput = zeros(self.n_outputs, **dev)
        #if self.ODE_order:
        #    lastoutput = zeros(self.n_outputs, **dev)


        #assert False, f'lastinput {lastinput.size()} lastinput {self.X_val.size()}'

        inputs = vstack([lastinput, self.X_val_extended]).view(-1, self.X_val_extended.shape[1])
        
        states = zeros((1, self.n_nodes), **dev)
        states[0,:] = laststate

        states_dot = zeros((1, self.n_nodes), **dev)
        #states_dot[0,:] = lasthdot

        outputs = lastoutput.view(1, self.n_outputs)
        

        #dt = inputs[1,:] - inputs[0,:]

        with no_grad():
            
            for t in range(n_samples):

                state_t, _  = self.train_state(t, X = inputs[t, :],
                                          state = states[t,:], y = None,#outputs[t,:],
                                         output = False) #y = outputs[t, :],
                states = cat([states, state_t.view(-1, self.n_nodes)], axis = 0)

                extended_state_spec = cat([inputs[t, :], states[t, :]]).view(-1, 1)#self.n_outputs)
                
                output_t = self.LinOut(extended_state_spec.T)
                    
                outputs = cat([outputs, output_t.view(1, self.n_outputs)], axis = 0)

            #drop first state
            self.val_states = states = states[1:]
            outputs = outputs[1:]

            #if self.burn_in:
            #    states = states[self.burn_in:]
            #    outputs = outputs[self.burn_in:]

            updates = self.LinIn(self.X_val) + self.bias + self.LinRes(states)
            states = torch.cat((self.X_val, states), axis = 1)
            time = self.X_val

            # calculate hidden state derivatives
            if self.ODE_order:
                states_dot = - self.alpha * states[:,1:] + self.alpha * self.activation_function(updates)
                # if self.ODE_order == 2:
                #     states_dot2 = - self.alpha * states_dot + self.alpha * self.act_f_prime(updates) * (self.LinIn.weight.T + self.bias + self.LinRes(states_dot))
                #     states_dot2 = torch.cat((zeros_like(self.X_val), states_dot2), axis = 1)
                states_dot = torch.cat((ones_like(self.X_val), states_dot), axis = 1)
                assert states.shape == states_dot.shape

                G = self.reparam(self.X_val, order = self.ODE_order)
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
                assert type(self.reparam) != type(None), "you must input a reparam function with ODE"
                
                if not self.multiple_ICs:
                    for i in range(self.ODE_order):
                        A_i = self.init_conds[i] * g.pow(i) #self.ode_coefs[i] * 
                        if not i:
                            A = A_i # + ode_coefs[1] * v0 * g.pow(1) + ... + ode_coefs[m] * accel_0 * g.pow(m)
                        else:
                            A = A + A_i
                else:
                    As = [y0 * g.pow(0) for y0 in self.init_conds[0]]

                if not self.multiple_ICs:
                    if not self.eq_system:
                        y = A + g * N
                        ydot = g_dot * N +  g * N_dot
                        N = self.LinOut(states)
                        #self.N = self.N.view(-1, self.n_outputs)
                        #gH = g * states
                        #gH_dot =  g_dot * states  +  g * states_dot

                        N_dot = self.calc_Ndot(states_dot, cutoff = False)
                        #elf.yfit = init_conds[0] + g.pow(1) * self.N
                        #self.ydot = g_dot * self.N +  g * N_dot
                    else:
                        y = g*self.LinOut(states)
                        #N = g*states
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
                            N_dot = self.calc_Ndot(states_dot, cutoff = False)
                            yfit = g*N
                            #for i, cond in enumerate(self.init_conds):
                            yfit[:, 0] = yfit[:,0] + self.init_conds[0][i]
                            #yfit[:, 1] = yfit[:,1] + init_conds[1]#[i]
                            
                            ydot = g_dot * N +  g * N_dot

                            ys.append(yfit)
                            ydots.append(ydot)

                        return ys, ydots
                        #y = A + g * N
                        
                        #elf.yfit = init_conds[0] + g.pow(1) * self.N
                        #self.ydot = g_dot * self.N +  g * N_dot
                    else:
                        ys = []
                        ydots = []
                        for i, weight in enumerate(self.weights_list):
                            #print("w", i)
                            self.LinOut.weight = Parameter(weight.view(self.n_outputs, -1))
                            self.LinOut.bias = Parameter(self.biases_list[i].view(1, self.n_outputs))

                            N = self.LinOut(states)
                            N_dot = self.calc_Ndot(states_dot, cutoff = False)

                            y = g*N

                            ydot = g_dot * N + N_dot * g
                            y[:, 0] = y[:,0] + self.init_conds[0][i]
                            y[:, 1] = y[:,1] + self.init_conds[1]
                            # for i, cond in enumerate(self.init_conds):
                            #     y[:, i] = y[:,i] + cond

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
            else:
                try:
                    if scale:
                        yhat = self.descale(outputs = outputs).view(-1, self.n_outputs) 
                except:
                    yhat = outputs
                return yhat
        #https://towardsdatascience.com/in-place-operations-in-pytorch-f91d493e970e



    # def predict_stepwise(self, y, x=None, steps_ahead=1, y_start=None):
    #     """Predicts a specified number of steps into the future for every time point in y-values array.
    #     E.g. if `steps_ahead` is 1 this produces a 1-step ahead prediction at every point in time.
    #     Parameters
    #     ----------
    #     y : numpy array
    #         Array with y-values. At every time point a prediction is made (excluding the current y)
    #     x : numpy array or None
    #         If prediciton requires inputs, provide them here
    #     steps_ahead : int (default 1)
    #         The number of steps to predict into the future at every time point
    #     y_start : float or None
    #         Starting value from which to start prediction. If None, last stored value from training will be used
    #     Returns
    #     -------
    #     y_predicted : numpy array
    #         Array of predictions at every time step of shape (times, steps_ahead)
    #     """

    #     # Check if ESN has been trained
    #     if self.out_weights is None or self.y_last is None:
    #         raise ValueError('Error: ESN not trained yet')

    #     # Normalize the arguments (like was done in train)
    #     y = self.scale(outputs=y)
    #     if not x is None:
    #         x = self.scale(inputs=x)

    #     # Timesteps in y
    #     t_steps = y.shape[0]

    #     # Check input
    #     if not x is None and not x.shape[0] == t_steps:
    #         raise ValueError('x has the wrong size for prediction: x.shape[0] = {}, while y.shape[0] = {}'.format(
    #             x.shape[0], t_steps))

    #     # Choose correct input
    #     if x is None and not self.feedback:
    #         #pass #raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
    #         inputs = ones((t_steps + steps_ahead, 2), **dev) 
    #     elif not x is None:
    #         # Initialize input
    #         inputs = ones((t_steps, 1), **dev)  # Add bias term
    #         inputs = hstack((inputs, x))  # Add x inputs
    #     else:
    #         # x is None
    #         inputs = ones((t_steps + steps_ahead, 1), **dev)  # Add bias term
        
    #     # Run until we have no further inputs
    #     time_length = t_steps if x is None else t_steps - steps_ahead + 1

    #     # Set parameters
    #     y_predicted = zeros((time_length, steps_ahead), dtype=self.dtype, device=self.device)

    #     # Get last states
    #     previous_y = self.y_last
    #     if not y_start is None:
    #         previous_y = self.scale(outputs=y_start)[0]

    #     # Initialize state from last availble in train
    #     current_state = self.state[-1]

    #     # Predict iteratively
    #     with no_grad():
            
    #         for t in range(time_length):

    #             # State_buffer for steps ahead prediction
    #             prediction_state = current_state.clone().detach()
                
    #             # Y buffer for step ahead prediction
    #             prediction_y = previous_y.clone().detach()
            
    #             # Predict stepwise at from current time step
    #             for n in range(steps_ahead):
                    
    #                 # Get correct input based on feedback setting
    #                 prediction_input = inputs[t + n] if not self.feedback else hstack((inputs[t + n], prediction_y))
                    
    #                 # Update
    #                 prediction_update = self.activation_function(matmul(self.in_weights, prediction_input.T) + 
    #                                                matmul(self.weights, prediction_state))
                    
    #                 prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state
                    
    #                 # Store for next iteration of t (evolves true state)
    #                 if n == 0:
    #                     current_state = prediction_state.clone().detach()
                    
    #                 # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
    #                 prediction_row = hstack((prediction_input, prediction_state))
    #                 if not self.backprop:
    #                     y_predicted[t, n] = matmul(prediction_row, self.out_weights)
    #                 else:
    #                     y_predicted[t, n] = self.LinOut.weight.T @ prediction_row[1:]
    #                 prediction_y = y_predicted[t, n]

    #             # Evolve true state
    #             previous_y = y[t]

    #     # Denormalize predictions
    #     y_predicted = self.descale(outputs=y_predicted)
        
    #     # Return predictions
    #     return y_predicted
    
    

    def descale(self, inputs=None, outputs=None):
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
        
        #for tensor in [train_x, train_y]:
        #     print('device',tensor.get_device())
        
        if not inputs is None:
            if self.ODE_order:
                transformed.append(inputs * self._input_stds)
            else:
                transformed.append((inputs * self._input_stds) + self._input_means)

        if not outputs is None:
            if self.ODE_order:
                transformed.append(outputs)
            else:
                transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

"""
                                    #calculate F depending on the order of the ODE:
                                    if ODE == 1:
                                        #population eq
                                        #RHS = lam * self.y0 - f(self.X) 
                                        #self.F =  g_dot * states_  +  g * (states_dot + lam * states_)
                                        
                                        #nl eq 
                                        self.F =  g_dot * states_  +  g * states_dot
                                        if nl_f:
                                            #y0_nl, y0_nl_dot = nl_f(self.y0)
                                            self.F = self.F - 2 * self.y0 * g * states_ 
                                            #self.F = self.F - (g * ).T @ 

                                    elif ODE == 2:
                                        # without a reparameterization
                                        #self.F = torch.square(self.X) * states_dot2 + 4 * self.X * states_dot + 2 * states_ + (self.X ** 2) * states_
                                        self.G = g * states_
                                        assert self.G.shape == states_.shape, f'{self.shape} != {self.states_.shape}'
                                        self.Lambda = g.pow(2) * states_ 
                                        self.k = 2 * states_ + g * (4*states_dot - self.G*states_) + g.pow(2) * (4 * states_ - 4 * states_dot + states_dot2)
                                        self.F = self.k + self.Lambda
                                    #common F derivation:
                                    F = self.F.T
                                    F1 = F.T @ F 
                                    F1 = F1 + self.regularization * eye(F1.shape[1], **self.dev)
                                    ##################################### non-linear adustment
                                    nl_adjust = False
                                    if nl_adjust:
                                        G = g * states_
                                        G_sq = G @ G.T
                                        nl_correction = -2 * self.y0 * (G_sq)
                                        F1 = F1 + nl_correction
                                    #F1_inv = pinverse(F1)
                                    #F2 = matmul(F1_inv, F.T)
                                    #####################################
                                    #First Order equation
                                    if self.ODE_order == 1:
                                        self.y0I = (self.y0 ** 2) * ones_like(self.X)
                                        #self.y0I = self.y0I.squeeze().unsqueeze(0)
                                        #RHS = lam*self.y0I.T - f(self.X) 

                                        #REPARAM population
                                        #RHS = lam * self.y0 - f(self.X) 

                                        RHS = self.y0I

                                        #weight = matmul(-F2.T, RHS)

                                        weight = matmul(F2.T, RHS)
                                        #assert False, weight.shape

                                    #Second Order equation
                                    elif self.ODE_order == 2:
                                        
                                        #self.y0I = y0[0] * ones_like(self.X)
                                        #self.y0I = self.y0I.squeeze().unsqueeze(0)

                                        #RHS = self.y0I.T + self.X * y0[1]
                                        RHS = self.y0 + f_t * y0[1]
                                        
                                        #t = self.X
                                        #A0 = y0 + g * v0
                                        #RHS = A0 + (g - 1)*v0 - f(t)
                                        weight = matmul(-F2.T, D_A)
                                    weight = matmul(D_W, D_A)

                                    #y = y0[0] + self.X * y0[1] + self.X
"""
"""
        if self.ODE_order == 1:
            return self.reparam(t = self.X, init_conditions = self.y0, N = self.yfit, N_dot = N_dot)
        elif self.ODE_order == 2:
            N_dot2 = self.calc_hdot(states_dot2[:,1:], cutoff = False)
            return self.reparam(t = self.X, init_conditions = [y0, v0], 
                N = self.yfit, N_dot = [N_dot, N_dot2], esn = self, 
                states = states_[:,1:], states_dot = states_dot[:,1:], states_dot2 = states_dot2[:,1:])
        """
"""
    #assert weight.requires_grad, "weight doesn't req grad"
                            #torch.solve solves AX = B. Here X is beta_hat, A is ridge_x, and B is ridge_y
                            #weight = torch.solve(ridge_y, ridge_x).solution
                        # elif self.l2_prop == 1:
                        # else: #+++++++++++++++++++++++         This section is elastic net         +++++++++++++++++++++++++++++++

                        #     gram_matrix = matmul(train_x.T, train_x) 

                        #     regr = ElasticNet(random_state=0, 
                        #                           alpha = self.regularization, 
                        #                           l1_ratio = 1-self.l2_prop,
                        #                           selection = "random",
                        #                           max_iter = 3000,
                        #                           tol = 1e-3,
                        #                           #precompute = gram_matrix.numpy(),
                        #                           fit_intercept = True
                        #                           )
                        #     print("train_x", train_x.shape, "_____________ train_y", train_y.shape)
                        #     regr.fit(train_x.numpy(), train_y.numpy())

                        #     weight = tensor(regr.coef_, device = self.device, **self.dev)
                        #     bias = tensor(regr.intercept_, device =self.device, **self.dev)
"""
                                    