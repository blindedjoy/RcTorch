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
from torch.quasirandom import SobolEngine
from torch import matmul, pinverse, hstack, eye, ones, zeros, cuda, Generator, rand, randperm, no_grad, normal, tensor, vstack, cat, dot
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

#pytorch elastic net regularization:
#https://github.com/jayanthkoushik/torch-gel

#TODO: unit test setting interactive to False.

#TODO: repair esn documentation (go strait to reinier's, copy and make adjustments)

#TODO: rename some pyesn variables.

#TODO: improve documentation.


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
                 regularization = None, activation_f = Tanh(), feedback = True,                              #<-- activation, feedback
                 input_scaling = 0.5, feedback_scaling = 0.5, noise = None,                                     #<-- hyper-params not needed for the hw
                 approximate_reservoir = False, device = None, id_ = None, random_state = 123, reservoir = None, #<-- process args
                 backprop = False, classification = False, criterion = NLLLoss(),  epochs = 7, l2_prop = 1): #<-- this line is backprop arguments
        super().__init__()
        
        #activation function
        self.activation_function = activation_f

        #cuda (gpu)
        if not device:
            self.device = torch_device("cuda" if cuda_is_available() else "cpu")
        else:
            self.device = device
        self.dtype = torch.float32

        # random state and default tensor arguments
        self.random_state = Generator(device=self.device).manual_seed(random_state)
        self.tensor_args = {"device": self.device, "generator" : self.random_state}

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

        #Feedback
        self.feedback = feedback

        #For speed up: approximate implimentation and preloaded reservoir matrices.
        self.approximate_reservoir, self.reservoir = approximate_reservoir, reservoir
        
        #backpropogation attributes:
        self.backprop, self.epochs = backprop, epochs

        #elastic net attributes: (default is 1, which is ridge regression for speed)
        self.l2_prop = l2_prop

        self.id_ = id_
        
        #Reservoir layer
        self.LinRes = Linear(self.n_nodes, self.n_nodes, bias = False)

        #https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        self.classification = classification
        if self.classification:
            self.log_reg = Linear(self.n_nodes, 2)
            self.criterion = criterion #torch.nn.CrossEntropyLoss()
        else:
            self.criterion = MSELoss()
            
        with no_grad():
            self.gen_reservoir()
        
        scaler = "standardize"
        if scaler == "standardize":
            self.scale   = self.normalize
            self.descale = self.denormalize

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

    def forward(self, t, input_, current_state, output_pattern):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.
        """
        # generator = self.random_state, device = self.device)  
        #assert 3 == 0, f'LinRes {self.LinRes(current_state).shape},'

        preactivation = self.LinIn(input_) + self.bias_ + self.LinRes(current_state)

        if self.feedback:
            preactivation += self.LinFeedback(output_pattern)
        
        #alternative: uniform noise
        #self.noise = rand(self.n_nodes, **self.tensor_args).view(-1,1) if noise else None

        update = self.activation_function(preactivation) # + self.PyESNnoise * (self.external_noise - 0.5)
        if self.noise != None:
            noise_vec = torch.normal(mean = torch.zeros(self.n_nodes, device = self.device), 
                                          std = torch.ones(self.n_nodes, device = self.device),
                                          generator = self.random_state)* self.noise
            update += noise_vec 
        next_state = self.leaking_rate[0] * update + self.leaking_rate[1] * current_state
        return next_state


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
            self.state = zeros(1, self.n_nodes, device=torch_device(self.device))
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
                bias = bias = zeros(n, 1, device = self.device)
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


    def train(self, y, X=None, burn_in=0, input_weight=None, verbose = False , learning_rate = None, return_states = False):
        """
        Train the network.
        
        Arguments: TODO
            y: response matrix
            x: observer matrix
            burn in: obvious
            input_weight : ???
            learning_rate: 
            verbose:
        """
        if type(y) == np.ndarray:
             y = torch.tensor(y, device = self.device, dtype = self.dtype, requires_grad = False)
        if y.device != self.device:
            y = y.to(self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
            
        if type(X) != None:
            if type(X) == np.ndarray:
                X = torch.tensor(X, device = self.device, dtype = self.dtype, requires_grad = False)
            if X.device != self.device:
                X = X.to(self.device)
            if len(X.shape) == 1:
                X = X.view(-1, 1)


        start_index = 1 if self.feedback else 0 
        rows = y.shape[0] - start_index
        
        # Normalize inputs and outputs
        y = self.scale(outputs=y, keep=True)
        
        try:
            orig_X = X.clone().detach()
        except:
            if not X:
                X = ones(*y.shape, device = self.device)
                orig_X = X.clone().detach()

        if not X is None:

            if X.std() != 0:

                X = self.scale(inputs=X, keep=True)
            else:
                self._input_stds = None
                self._input_means = None
        
        self.n_outputs = y.shape[1]
        self.n_inputs = X.shape[1]
        
        self.lastoutput = y[-1, :]
        self.lastinput = X[-1, :]

        self.n_outputs  = y.shape[1]
        
        start_index = 1 if self.feedback else 0 
        rows = y.shape[0] - start_index

        self.LinIn = Linear(self.n_nodes, self.n_inputs, bias = False)
        self.LinFeedback = Linear(self.n_nodes, self.n_inputs, bias = False)
        self.LinIn.weight, self.LinFeedback.weight = self.set_Win()
        self.LinOut = Linear(self.n_nodes + 1, self.n_outputs)
        if not self.classification:
            self.out_activation = self.LinOut
        
        #build the state matrix:
        self.state = zeros((X.shape[0], self.n_nodes), device = self.device).detach()
        self.state._name_ = "state"

        current_state = self.state[-1] 


        if not self.backprop:
            self.LinOut.requires_grad_, self.LinIn.requires_grad_, self.LinFeedback.requires_grad_ = False, False, False

        self.burn_in = burn_in

        self.losses = None
            
        #fast exact solution ie we don't want to run backpropogation (ie we aren't doing classification):
        if not self.backprop:
            with no_grad():

                #run through the states.
                for t in range(1, X.shape[0]):
                    self.state[t, :] = self.forward(t, input_ = X[t, :].T,
                                                       current_state = self.state[t-1,:], 
                                                       output_pattern = y[t-1]).squeeze()

                #print("self.state", self.state)
                #print("X", X)
                #print("ones", ones(*X.shape))
                extended_states = hstack((self.state, X))
                extended_states._name_ = "complete_data"

                train_x = extended_states[burn_in:, :]
                train_y = y[burn_in:]
                bias = None
                if not self.regularization:
                    print("no regularization")
                    pinv = pinverse(train_x)
                    weight = matmul(pinv, train_y)
                elif self.l2_prop == 1:
                    #print("ridge regularizing")
                    ones_col = ones(train_x.shape[0], 1, device = self.device)
                    train_x = hstack((ones_col, train_x))
                    
                    ridge_x = matmul(train_x.T, train_x) + \
                                       self.regularization * eye(train_x.shape[1], device = self.device)
                    ridge_y = matmul(train_x.T, train_y)
                    ridge_x_inv = pinverse(ridge_x)
                    weight = ridge_x_inv @ ridge_y

                    bias = weight[0]
                    weight = weight[1:]
                    #torch.solve solves AX = B. Here X is beta_hat, A is ridge_x, and B is ridge_y
                    #weight = torch.solve(ridge_y, ridge_x).solution

                else:

                    gram_matrix = matmul(train_x.T, train_x) 

                    regr = ElasticNet(random_state=0, 
                                          alpha = self.regularization, 
                                          l1_ratio = 1-self.l2_prop,
                                          selection = "random",
                                          max_iter = 3000,
                                          tol = 1e-3,
                                          #precompute = gram_matrix.numpy(),
                                          fit_intercept = True
                                          )
                    print("train_x", train_x.shape, "_____________ train_y", train_y.shape)
                    regr.fit(train_x.numpy(), train_y.numpy())

                    weight = tensor(regr.coef_, device = self.device)
                    bias = tensor(regr.intercept_, device =self.device)
                
                self.LinOut.weight = Parameter(weight.view(-1,1), requires_grad = False)
                if type(bias) != type(None):
                    self.LinOut.bias = Parameter(bias.view(-1,1), requires_grad = False)
                
                self.laststate = self.state[-1, :]
        else:
            # backprop:
            print(5)
            running_loss = 0
            train_losses = []
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            for e in range(self.epochs):
                optimizer.zero_grad()
                loss = 0
                #unnin
                for t in range(inputs.shape[0]):

                    input_ = inputs[t].T
                    _, output = self.forward(t, input_, current_state)
                    
                    loss += self.criterion(output.view(-1,), y[t])
                    if t % 500 == 0:
                        print("timestep ", t)
                if not e:   
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                optimizer.step()

                ################################################## Classification, from Marios's prototype
                #running_loss += loss.item()
                #if self.classification:
                #    output = F.log_softmax(output, dim=1)
                #   #z_targ = y#.view(-1,) #z_targ = z_targ.long()
                #    running_loss += self.criterion(output, y) 
                #else:
                #    print(self.criterion)
                #    running_loss += self.criterion(output, y)
                #loss_history.append(loss.data.numpy())
                ##################################################

                #print('Linout weights', self.LinOut.weight.shape)
                #print('Linout bias', self.LinOut.bias.shape)
                

                print("Epoch: {}/{}.. ".format(e+1, self.epochs),
                      "Training Loss: {:.3f}.. ".format(loss))#/len(trainloader)),
                      #"Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      #"Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            self.out_weights = self.LinOut.weight
            self.out_weights._name_ = "out_weights"
            complete_data = hstack((inputs, self.state))
        
        # Store last y value as starting value for predictions
        self.y_last = y[-1, :]

        # Return all data for computation or visualization purposes (Note: these are normalized)
        if return_states:
            return extended_states, (y[1:,:] if self.feedback else y), burn_in
        else:
            yfit_norm = self.LinOut.weight.T.cpu()@extended_states.T.cpu() + self.LinOut.bias.cpu()
            yfit = self._output_stds.cpu()* (yfit_norm)+ self._output_means.cpu()
            return yfit.detach().numpy()
    
    def normalize(self, inputs=None, outputs=None, keep=False):
        """Normalizes array by column (along rows) and stores mean and standard devation.

        Set `store` to True if you want to retain means and stds for denormalization later.

        Parameters
        ----------
        inputs : array or None
            Input matrix that is to be normalized
        outputs : array or No ne
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
                self._input_stds = inputs.std(dim = 0)

            # Transform
            transformed.append((inputs - self._input_means) / self._input_stds)

        if not outputs is None:
            if keep:
                # Store for denormalization
                self._output_means = outputs.mean(axis=0)
                self._output_stds = outputs.std(dim = 0)#, ddof=1)

            # Transform
            transformed.append((outputs - self._output_means) / self._output_stds)
            
            self._output_means = self._output_means
            self._output_stds = self._output_stds
        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

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
        self.steps_ahead = steps_ahead

        if type(y) == np.ndarray:
             y = torch.tensor(y, device = self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        if y.device != self.device:
            y = y.to(self.device)
        if type(x) != None:
            if type(x) == np.ndarray:
                x = tensor(x, device = self.device)
            if x.device != self.device:
                x = x.to(self.device)

        
        # Run prediction
        final_t =y.shape[0]
        if steps_ahead is None:
            if x is None:
                x = ones(*y.shape, device = self.device)
            y_predicted = self.predict(n_steps = y.shape[0], x=x, y_start=y_start)
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            y_predicted = self.predict_stepwise(y, x, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]
        
        score = self.error(y_predicted, y, scoring_method, alpha=alpha)
        
        # Return error
        if self.id_ == None:
            #user friendly
            return float(score), y_predicted.cpu().numpy()
        else:
            #internal to esn_cv
            return score, y_predicted, self.id_


    def predict(self, n_steps, x=None, y_start=None, continuation = True):
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
        if self.y_last is None: 
            raise ValueError('Error: ESN not trained yet')
        
        # Normalize the inputs (like was done in train)
        if not x is None and type(self._input_means) != type(None):
            x = self.scale(inputs=x)

        dev = {"device" : self.device, "dtype" : torch.float32}

        
        # Set parameters
        if self.LinOut.weight.shape[0] == 1:
            y_predicted = zeros( (n_steps,), **dev)
        else:
            y_predicted = zeros( (n_steps, self.LinOut.weight.shape[0]), **dev)

        n_samples = x.shape[0]

        if not y_start is None: #if not x is None:
            previous_y = self.scale(outputs=y_start)[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = zeros(self.n_nodes, **dev)
            lastinput = zeros(self.n_inputs, **dev)
            lastoutput = zeros(self.n_outputs, **dev)

        inputs = vstack([lastinput, x]).view(-1, x.shape[1])
        states = zeros((n_samples + 1, self.n_nodes), **dev)
        states[0,:] = laststate

        outputs = vstack(
            [lastoutput, zeros((n_samples, self.n_outputs), **dev)])

        for t in range(n_samples):
            states[t + 1, :] = self.forward(t, input_ = inputs[t + 1, :], 
                                                current_state = states[t, :], 
                                                output_pattern = outputs[t, :])
            extended_state_spec = cat([states[t+1,:], inputs[t+1, :]])
            #print("extended_state_spec", extended_state_spec.shape)
            #print("self.LinOut.weight", self.LinOut.weight.shape)
            #print("self.LinOut.bias", self.LinOut.bias.shape)
            bias_tensor = ones(*self.LinOut.weight.shape).squeeze()*self.LinOut.bias.item()
            #print("Linout", self.LinOut.weight.shape)
            #print("bias_tensor", bias_tensor.shape)
            #print("extended_state_spec", extended_state_spec.shape)
            outputs[t+1,:] = dot(self.LinOut.weight.squeeze(),extended_state_spec.squeeze())
            #print("outputs[t+1,:].shape", outputs[t+1,:].shape)
            #print("bias_tensor", bias_tensor.shape)
            outputs[t+1,:] = outputs[t+1,:] + float(self.LinOut.bias) #torch.dot(self.LinOut.weight.squeeze(), extended_state_spec.squeeze()) + self.LinOut.bias#torch.dot(self.LinOut.weight, 

        return self.denormalize(outputs = outputs[1:]).view(-1, self.n_outputs) 
        try:
            return self.denormalize(outputs = outputs[1:]).view(-1, self.n_outputs) # 
        except:
            return outputs[1:]



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
        y = self.scale(outputs=y)
        if not x is None:
            x = self.scale(inputs=x)

        # Timesteps in y
        t_steps = y.shape[0]

        # Check input
        if not x is None and not x.shape[0] == t_steps:
            raise ValueError('x has the wrong size for prediction: x.shape[0] = {}, while y.shape[0] = {}'.format(
                x.shape[0], t_steps))

        # Choose correct input
        if x is None and not self.feedback:
            #pass #raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
            inputs = ones((t_steps + steps_ahead, 2), dtype=torch.float32, device = self.device) 
        elif not x is None:
            # Initialize input
            inputs = ones((t_steps, 1), dtype=torch.float32)  # Add bias term
            inputs = hstack((inputs, x))  # Add x inputs
        else:
            # x is None
            inputs = ones((t_steps + steps_ahead, 1), dtype=torch.float32, device = self.device)  # Add bias term
        
        # Run until we have no further inputs
        time_length = t_steps if x is None else t_steps - steps_ahead + 1

        # Set parameters
        y_predicted = zeros((time_length, steps_ahead), dtype=torch.float32, device=self.device)

        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.scale(outputs=y_start)[0]

        # Initialize state from last availble in train
        current_state = self.state[-1]

        # Predict iteratively
        with no_grad():
            
            for t in range(time_length):

                # State_buffer for steps ahead prediction
                prediction_state = current_state.clone().detach()
                
                # Y buffer for step ahead prediction
                prediction_y = previous_y.clone().detach()
            
                # Predict stepwise at from current time step
                for n in range(steps_ahead):
                    
                    # Get correct input based on feedback setting
                    prediction_input = inputs[t + n] if not self.feedback else hstack((inputs[t + n], prediction_y))
                    
                    # Update
                    prediction_update = self.activation_function(matmul(self.in_weights, prediction_input.T) + 
                                                   matmul(self.weights, prediction_state))
                    
                    prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state
                    
                    # Store for next iteration of t (evolves true state)
                    if n == 0:
                        current_state = prediction_state.clone().detach()
                    
                    # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
                    prediction_row = hstack((prediction_input, prediction_state))
                    if not self.backprop:
                        y_predicted[t, n] = matmul(prediction_row, self.out_weights)
                    else:
                        y_predicted[t, n] = self.LinOut.weight.T @ prediction_row[1:]
                    prediction_y = y_predicted[t, n]

                # Evolve true state
                previous_y = y[t]

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

        #most loss functions not implimented
        # Compute mean error
        if method == 'mse':
            error = torch.mean(torch.square(errors))
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
        return error.type(torch.float32)
    
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
        
        #for tensor in [train_x, train_y]:
        #     print('device',tensor.get_device())
        
        if not inputs is None:
            transformed.append((inputs * self._input_stds) + self._input_means)

        if not outputs is None:
            transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

    def destandardize(self, inputs=None, outputs=None):
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
            transformed.append((inputs * self._input_stds) + self._input_means)

        if not outputs is None:
            transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]

#TODO consider removing this. we're using sklearn after all
class ElasticNetRegularization(nn.Module):
    """TODO

    Arguments: TODO
    """
    def __init__(self, iterations, l2_proportion, regularization_parameter, learning_rate = None,
                       fail_tolerance = 10, val_prop = 0.2, ridge_weights = None, scaler = "normalize"):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_prop= l2_proportion
        self.reg_param = regularization_parameter
        self.val_prop = val_prop
        self.fail_tolerance = fail_tolerance
        self.ridge_weights  = ridge_weights

        assert scaler in ["normalize", "standardize"]
        if scaler == "standardize":
            self.scale   = self.standardize
            self.descale = self.destandarize
        #else:
        #    self.scale = self.standardize
        #    self.descale = self.destandardize
    #TODO: combine standardize and normalize into a class.
    
    def standardize(self, inputs = None, outputs = None, keep = False):
        """TODO

        Arguments: TODO
        """

        if type(inputs) != type(None):
            if keep:
                self.input_stds  = inputs.std(axis = 0)
                self.input_means  = inputs.mean(axis = 0)

            std_tensor = (inputs - self.input_means)/self.input_stds
        if type(outputs) != type(None):
            if keep:
                self.output_means = outputs.mean(axis = 0)
                self.output_stds  = outputs.std(axis = 0)
            std_tensor = (outputs - self.output_means)/self.output_stds
        
        return normalized_tensor

    def normalize(self, inputs = None, outputs = None, keep = False):
        
        """TODO

        Arguments: TODO
        """

        if type(inputs) != type(None):
            if keep:
                self.input_min  = inputs.min(axis = 0)
                self.input_max  = inputs.max(axis = 0)
            normalized_tensor = (inputs - self.input_min)/(self.input_max - self.input_min)
        if type(outputs) != type(None):
            if keep:
                self.output_min = outputs.min(axis = 0)
                self.output_max = outputs.max(axis = 0)
            normalized_tensor = (outputs - self.output_min)/(self.input_max - self.input_min)
        
        return std_tensor
    
    def denormalize(self, inputs = None, outputs = None):
        """TODO

        Arguments: TODO
        """
        assert 1 ==0, "not properly implimented"
        if type(inputs) != type(None):
            denormalized_tensor = (inputs +self.input_means)*self.input_stds
        if type(outputs) != type(None):
            denormalized_tensor = (outputs + self.output_means)*self.output_stds
        return denormalized_tensor
    
    def elastic_net_loss(self, output, target):
        """TODO

        Arguments: TODO
        """
        l1_loss_component = (1 - self.l2_prop) * torch.sum(torch.abs(self.linear.weight))
        if type(self.ridge_weights) == type(None):
            l2_loss_component = self.l2_prop * torch.sum(torch.square(self.linear.weight))
        else:
            l2_loss_component = self.l2_prop * torch.sum(torch.square(self.linear.weight - self.ridge_weights))
        loss = torch.sum(torch.square(output - target)) +  self.reg_param * (l1_loss_component + l2_loss_component)
        return loss
    
    def fit(self, X, y):
        """TODO

        Arguments: TODO
        """
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        
        with torch.enable_grad():
            X = self.scale(inputs = X, keep = True)
            y = self.scale(outputs = y, keep = True)

        if not X.requires_grad:
            X.requires_grad = True
        if not y.requires_grad:
            y.requires_grad = True

        if not X.requires_grad:
            assert 1 == 0
        if not y.requires_grad:
            assert 1 == 0
        
        self.linear = Linear(self.n_features, 1)
        
        #gradient descent learning
        if self.learning_rate:
            optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        else:
            
            optimizer = optim.Adam(self.parameters())
        loss = 0
        
        self.losses = []
        self.fails = []
        for e in range(self.iterations):
            val_idx= randperm(int(self.n_samples*(self.val_prop)))
            train_X = X[~val_idx,:]
            train_X = self.scale(inputs = train_X, keep = False)
            
            prediction = self.forward(train_X)
            if not prediction.requires_grad:
                prediction.requires_grad = True
            loss = self.elastic_net_loss(prediction, y[~val_idx,:])
            if not e:   
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.losses.append(loss.detach())
            if e >= 1:
                self.fails.append(self.losses[-2] < self.losses[-1])
            if e > (self.fail_tolerance + 20):
                n_fails =  sum(self.fails[-self.fail_tolerance:]) 
                if n_fails == self.fail_tolerance:
                    print(n_fails)
                    break
            optimizer.step()

                
        return self.linear, self.losses
    
    def forward(self, X):
        return self.linear(X)
    
    def predict(self, X):
        X = self.scale(inputs = X, keep = False)

        with no_grad():
            pred = self.forward(X)
        
        return self.denormalize(outputs = pred).detach()
