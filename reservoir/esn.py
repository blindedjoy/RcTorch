#Imports
import math
from dataclasses import dataclass

#botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

#gpytorch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd import grad
from torch.quasirandom import SobolEngine

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

dtype=torch.float 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print("device:", device)

torch.autograd.set_detect_anomaly(True)

def printn(param: torch.nn.parameter):
    print(param._name_ + "\t \t", param.shape)

def NRMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2)/y**2)

def sinsq(x):
    return torch.square(torch.sin(x))

def printc(string_, color_) :
      print(colorz[color_] + string_ + colorz["endc"] )

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

class ElasticNetRegularization(nn.Module):
    def __init__(self, iterations, l2_proportion, regularization_parameter, learning_rate = None,
                       fail_tolerance = 10, val_prop = 0.2, ridge_weights = None):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_prop= l2_proportion
        self.reg_param = regularization_parameter
        self.val_prop = val_prop
        self.fail_tolerance = fail_tolerance
        self.ridge_weights  = ridge_weights
    
    def normalize(self, inputs = None, outputs = None, keep = False):
        
        if type(inputs) != type(None):
            if keep:
                self.input_means  = inputs.mean(axis = 0)
                self.input_stds  = inputs.std(axis = 0)
            normalized_tensor = (inputs - self.input_means)/self.input_stds
        if type(outputs) != type(None):
            if keep:
                self.output_means = outputs.mean(axis = 0)
                self.output_stds  = outputs.std(axis = 0)
            normalized_tensor = (outputs - self.output_means)/self.output_stds
        
        return normalized_tensor
    
    def denormalize(self, inputs = None, outputs = None):
        if type(inputs) != type(None):
            denormalized_tensor = (inputs +self.input_means)*self.input_stds
        if type(outputs) != type(None):
            denormalized_tensor = (outputs + self.output_means)*self.output_stds
        return denormalized_tensor
    
    def elastic_net_loss(self, output, target):
        l1_loss_component = (1 - self.l2_prop) * torch.sum(torch.abs(self.linear.weight))
        if type(self.ridge_weights) == type(None):
            l2_loss_component = self.l2_prop * torch.sum(torch.square(self.linear.weight))
        else:
            l2_loss_component = self.l2_prop * torch.sum(torch.square(self.linear.weight - self.ridge_weights))
        loss = torch.sum(torch.square(output - target)) +  self.reg_param * (l1_loss_component + l2_loss_component)
        return loss
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        
        with torch.enable_grad():
            X = self.normalize(inputs = X, keep = True)
            y = self.normalize(outputs = y, keep = True)

        if not X.requires_grad:
            X.requires_grad = True
        if not y.requires_grad:
            y.requires_grad = True

        if not X.requires_grad:
            assert 1 == 0
        if not y.requires_grad:
            assert 1 == 0
        
        self.linear = torch.nn.Linear(self.n_features, 1)
        
        #gradient descent learning
        if self.learning_rate:
            optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        else:
            
            optimizer = optim.Adam(self.parameters())
        loss = 0
        
        self.losses = []
        self.fails = []
        for e in range(self.iterations):
            val_idx= torch.randperm(int(self.n_samples*(self.val_prop)))
            train_X = X[~val_idx,:]
            train_X = self.normalize(inputs = train_X, keep = False)
            
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
        X = self.normalize(inputs = X, keep = False)

        with torch.no_grad():
            pred = self.forward(X)
        
        return self.denormalize(outputs = pred).detach()


class EchoStateNetwork(nn.Module):
    def __init__(self, spectral_radius=0.9, n_nodes = 1000, activation_f = nn.Tanh(), feedback = True,
                 noise = 0, input_scaling = 0.5, leaking_rate = 0.99, regularization = 10 **-3, backprop = False,
                 criterion = nn.NLLLoss(), classification = False, output_size = 50, feedback_scaling = 0.5,
                 already_normalized = False, bias = "uniform", connectivity = 0.1, random_state = 123,
                 exponential = False, obs_idx = None, resp_idx = None,
                 reservoir = None, model_type = "uniform", input_weight_type = None, approximate_reservoir = True,
                 device = device, epochs = 7, PyESNnoise=0.001, l2_prop = 1, reg_lr = 10**-4):
        super().__init__()
        
        self.l2_prop = l2_prop 
        self.reg_lr = reg_lr

        self.epochs = epochs

        #faster, approximate implimentation
        self.approximate_reservoir = approximate_reservoir
        self.reservoir = reservoir
        
        # is this obselete? check.
        self.already_normalized = already_normalized
        
        # backprop, feedback, random state and device ('cuda' or not)
        self.backprop = backprop
        self.device = torch.device(device)
        self.feedback = feedback
        self.random_state = torch.Generator(device=self.device).manual_seed(random_state)
        self.tensor_args = {"device": self.device, "generator" : self.random_state}
        
        # hyper-parameters:
        self.bias = bias
        self.connectivity = connectivity
        self.feedback_scaling = feedback_scaling
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.noise = noise
        self.n_nodes = n_nodes
        self.spectral_radius = spectral_radius
        self.regularization = regularization

        self.PyESNnoise = 0.001
        self.external_noise = torch.rand(self.n_nodes)

        
        #activation
        self.activation_function = activation_f
        
        #backprop layers
        self.LinRes = nn.Linear(self.n_nodes, self.n_nodes, bias = False)

        
        #https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        self.classification = classification
        if self.classification:
            self.log_reg = torch.nn.Linear(self.n_nodes, 2)
            self.criterion = criterion #torch.nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
            
        with torch.no_grad():
            self.gen_reservoir()
    
    def plot_reservoir(self):
        sns.histplot(self.weights.numpy().view(-1,))
        
    def forward(self, t, input_, current_state):
        """
        Arguments:
            t:
            input_:
            current_state:
        """
        #uses of torch.no_grad() are indiscriminant and potentially ineffective. improve this
        with torch.no_grad():
            in_vec = torch.matmul(self.in_weights, input_)

        weight_vec = torch.matmul(self.weights, current_state)

        update = self.activation_function(in_vec + weight_vec) 

        current_hidden_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state  # Leaking separate
        current_hidden_state = current_hidden_state.view(1,-1)
        
        with torch.no_grad():
            self.state[t, :] = current_hidden_state.detach().squeeze()
            
        if self.classification:
            output = self.ClassOut(current_hidden_state)
        else:
            if self.backprop:
                if self.feedback:
                    
                    vec = torch.cat([input_[1].view(-1,1), current_hidden_state], 1)

                    output = self.LinOut(vec)
                else:    
                    output = self.LinOut(current_hidden_state.view(-1,1))
            else:
                output = None
                #output = self.out_weights @ current_hidden_state
        
        return current_hidden_state, output #, hidden_dot, output_dot

    def forward2(self, t, input_, current_state, output_pattern, verbose = False):
        """
        Arguments:
            t: the current timestep
            input_: the input vector for timestep t
            current_state: the current state at timestep t
            output_pattern: the output pattern at timestep t.
        """

        preactivation = self.LinIn(input_) + self.LinRes(current_state)
        if self.feedback:
            preactivation += self.LinFeedback(output_pattern)

        update = self.activation_function(preactivation) + self.PyESNnoise * (self.external_noise - 0.5)

        return self.leaking_rate * update + (1 - self.leaking_rate) * current_state


    def gen_reservoir(self, obs_idx = None, targ_idx = None, load_failed = None):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state
        start = time.time()

        #random_state = np.random.RandomState(self.random_state)

        max_tries = 1000  # Will usually finish on the first iteration
        n = self.n_nodes

        #if the size of the reservoir has changed, reload it.
        if self.reservoir:
            if self.reservoir.n_nodes_ != self.n_nodes:
                load_failed = 1

        book_index = 0
        for i in range(max_tries):
            if i > 0:
                printc(str(i), 'fail')

            #only initialize the reservoir and connectivity matrix if we have to for speed in esn_cv.
            if not self.reservoir or not self.approximate_reservoir or load_failed == 1:

                self.accept = torch.rand(self.n_nodes, self.n_nodes, **self.tensor_args) < self.connectivity
                self.weights = torch.rand(self.n_nodes, self.n_nodes, **self.tensor_args) * 2 - 1
                self.weights *= self.accept
                #self.weights = csc_matrix(self.weights)
            else:
                #print("LOADING MATRIX", load_failed)
                if self.approximate_reservoir:
                
                    try:   
                        self.weights = self.reservoir.get_approx_preRes(self.connectivity, i).to(device)
                        #printc("reservoir successfully loaded (" + str(self.weights.shape) , 'green') 
                    except:
                        printc("approx reservoir " + str(i) + " failed to load... regenerating", 'fail')

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
                printc("Loaded Reservoir is Nilpotent (max_eigenvalue ={}), connectivity ={}.. .regenerating".format(max_eigenvalue, round(self.connectivity,8)), 'fail')
                
                #if we have run out of pre-loaded reservoirs to draw from :
                if i == max_tries - 1:
                    raise ValueError('Nilpotent reservoirs are not allowed. Increase connectivity and/or number of nodes.')

        # Set spectral radius of weight matrix
        self.weights = self.weights * self.spectral_radius / max_eigenvalue
        self.weights = nn.Parameter(self.weights, requires_grad = False)

        self.LinRes.weight = self.weights
        
        if load_failed == 1 or not self.reservoir:
            self.state = torch.zeros(1, self.n_nodes, device=torch.device(device))
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
        with torch.no_grad():
            
            if not self.reservoir or 'in_weights' not in dir(self.reservoir): 
                #print("GENERATING IN WEIGHTS")

                in_weights = torch.rand(self.n_nodes, self.n_inputs, generator = self.random_state, device = self.device)
                in_weights =  in_weights * 2 - 1
                
                if self.bias == "uniform":
                    #random uniform distributed bias
                    bias = torch.rand(self.n_nodes, 1, generator = self.random_state, device = self.device)
                    bias = bias * 2 - 1
                else:
                    bias = torch.ones(self.n_nodes, 1, device = self.device) * self.bias

                #if there is white noise add it in (this will be much more useful later with the exponential model)
                if self.noise:
                    white_noise = torch.normal(0, self.noise, device = self.device, size = (self.n_nodes, n_inputs))
                    in_weights += white_noise

                in_weights = torch.hstack((bias, in_weights)) * self.input_scaling
                
                if self.feedback:
                    feedback_weights = torch.rand(self.n_nodes, 1, device = self.device, generator = self.random_state) * 2 - 1
                    feedback_weights = self.feedback_scaling * feedback_weights
                    in_weights = torch.hstack((in_weights, feedback_weights)).view(self.n_nodes, -1)
                
            else:
                #print("self.reservoir.in_weights", self.reservoir.in_weights.get_device())
                #print("self.reservoir.noise_z", self.reservoir.noise_z.get_device())
                #print("feedback", self.reservoir.feedback_weights.get_device())
                in_weights = self.reservoir.in_weights + self.noise * self.reservoir.noise_z
                
                if self.feedback:
                    feedback_weights = self.feedback_scaling * self.reservoir.feedback_weights
                    in_weights = torch.hstack((in_weights, feedback_weights)).view(self.n_nodes, -1)
   
        in_weights = nn.Parameter(in_weights, requires_grad = False).to(self.device)
        in_weights._name_ = "in_weights"

        return(in_weights)

    def set_Win2(self): #inputs
        """
        Build the input weights.
        Currently only uniform implimented.

        Arguments:
            inputs:
        """
        with torch.no_grad():
            
            if not self.reservoir or 'in_weights' not in dir(self.reservoir): 
                print("GENERATING IN WEIGHTS")

                in_weights = torch.rand(self.n_nodes, self.n_inputs, generator = self.random_state, device = self.device)
                in_weights =  in_weights * 2 - 1
                
                if self.bias == "uniform":
                    #random uniform distributed bias
                    bias = torch.rand(self.n_nodes, 1, generator = self.random_state, device = self.device)
                    bias = bias * 2 - 1
                else:
                    bias = torch.ones(self.n_nodes, 1, device = self.device) * self.bias

                #if there is white noise add it in (this will be much more useful later with the exponential model)
                if self.noise:
                    # CURRENTLY THIS IS NOT BEING USED AT ALL
                    in_weight_white_noise = torch.normal(0, self.noise, device = self.device, size = (self.n_nodes, n_inputs))
                    #in_weights += white_noise

                in_weights = bias * self.input_scaling #torch.hstack((bias, in_weights)) * self.input_scaling

                """
                
                """
                
            else:
                
                in_weights = self.reservoir.in_weights  #+ self.noise * self.reservoir.noise_z One possibility is to add noise here, another is after activation.
                
                ##### Later for speed re-add the feedback weights here.

                #if self.feedback:
                #    feedback_weights = self.feedback_scaling * self.reservoir.feedback_weights
                #    in_weights = torch.hstack((in_weights, feedback_weights)).view(self.n_nodes, -1)

        if self.feedback:
            feedback_weights = torch.rand(self.n_nodes, 1, device = self.device, generator = self.random_state) * 2 - 1
            feedback_weights *= self.feedback_scaling
            feedback_weights = feedback_weights.view(self.n_nodes, -1)
            feedback_weights = nn.Parameter(feedback_weights, requires_grad = False) 
        else:
            feedback_weights = None
   
        in_weights = nn.Parameter(in_weights, requires_grad = False) #.to(self.device)
        #.to(self.device)
        #in_weights._name_ = "in_weights"

        return (in_weights, feedback_weights)
    
        
    def display_in_weights(self):
        sns.heatmap(self.in_weights)

    def display_out_weights(self):
        sns.heatmap(self.out_weights)

    def display_res_weights(self):
        sns.heatmap(self.weights)

    def plot_states(self, n= 10):
        for i in range(n):
            plt.plot(list(range(len(self.state[:,i]))), RC.state[:,i], alpha = 0.8)
        
    def train(self, y, x=None, burn_in=0, input_weight=None, verbose = False ,learning_rate = None):
        """
        Train the network.
        
        Arguments:
            y: response matrix
            x: observer matrix
            burn in: obvious
            input_weight : ???
            
        """
        #h  = Variable(torch.zeros([Nt,self.reservoir_size]), requires_grad = False)
        #zd = Variable(torch.zeros(Nt), requires_grad = False)
        #z  = Variable(torch.zeros(Nt), requires_grad = False)
        
        #TODO : torch random state
        
        
        #with torch.no_grad():
        #if x:
        #    x = x.to(device)
        #y = y.to(device)
        
        start_index = 1 if self.feedback else 0 
        rows = y.shape[0] - start_index
        
        # Normalize inputs and outputs
        y = self.normalize(outputs=y, keep=True)
        
        
        if not x is None:
            if not self.already_normalized:
                x = self.normalize(inputs=x, keep=True)
            self.n_inputs = (x.shape[1])

        self.lastoutput = y[-1, :]
        self.lastinput = x[-1, :]
    
        if x is None and not self.feedback:
            #raise ValueError("Error: provide x or enable feedback")
            self.already_normalized = True
            inputs = torch.ones((y.shape[0], y.shape[1] + 1), device = self.device)
            self.n_inputs  = y.shape[1]
        
        if x is None and self.feedback:
            self.n_inputs  = y.shape[1] - 1
        
        #build the state matrix:
        self.state = torch.zeros((rows, self.n_nodes), device = self.device)
        self.state._name_ = "state"

        #self.state = self.state.detach()
        current_state = self.state[-1] 
        
        #concatenate a column of ones to the input x for bias.
        inputs = torch.ones(rows, 1, device = self.device)
        inputs.requires_grad=False
        
        #initialize the in_weights and bias tensors
        if not x is None:
            inputs = torch.hstack((inputs, x[start_index:]))
        elif self.feedback:
            inputs = torch.hstack((inputs, y[:-1])) 
        #else:
        #    inputs = torch.rand(inputs.shape[0], 2, generator = self.random_state, device = self.device) * 2 -1#torch.hstack((inputs, inputs)) 

        self.in_weights = self.set_Win()#inputs)
        inputs._name_ = "inputs"

        self.burn_in = burn_in
            
        #fast exact solution ie we don't want to run backpropogation (ie we aren't doing classification):
        if not self.backprop:
            with torch.no_grad():

                for t in range(inputs.shape[0]):
                    hidden, output = self.forward(t, inputs[t].T, current_state)

                complete_data = torch.hstack((inputs, self.state))
                complete_data._name_ = "complete_data"

                train_x = complete_data[burn_in:]  # Include everything after burn_in
                train_y = y[burn_in + 1:] if self.feedback else y[burn_in:]

                # Ridge regression
                ridge_x = torch.matmul(train_x.T, train_x) + \
                            self.regularization * torch.eye(train_x.shape[1], device = self.device)
                ridge_y = torch.matmul(train_x.T, train_y)

                # Solver solution (fast)
                out_weights_sol = torch.solve(ridge_y, ridge_x)
                self.out_weights = out_weights_sol.solution
                self.out_weights._name_ = "out_weights"
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
            complete_data = torch.hstack((inputs, self.state))
        
        # Store last y value as starting value for predictions
        self.y_last = y[-1, :]

        # Return all data for computation or visualization purposes (Note: these are normalized)
        return complete_data, (y[1:,:] if self.feedback else y), burn_in

    def train2(self, y, x=None, burn_in=0, input_weight=None, verbose = False ,learning_rate = None):
        """
        Train the network.
        
        Arguments:
            y: response matrix
            x: observer matrix
            burn in: obvious
            input_weight : ???
            
        """

        start_index = 1 if self.feedback else 0 
        rows = y.shape[0] - start_index
        
        # Normalize inputs and outputs
        y = self.normalize(outputs=y, keep=True)
        
        orig_X = x.clone().detach()
        if not x is None:
            if x.std() != 0:
                x = self.normalize(inputs=x, keep=True)
            else:
                assert "invalid input, zero std"
            self.n_inputs = (x.shape[1])

        self.lastoutput = y[-1, :]
        self.lastinput = x[-1, :]

        self.n_inputs = 1#x.shape[1]
        self.n_outputs  = y.shape[1]
        
        start_index = 1 if self.feedback else 0 
        rows = y.shape[0] - start_index
        

        self.LinIn = nn.Linear(self.n_nodes, 1, bias = False).to(device)
        self.LinFeedback = nn.Linear(self.n_nodes, 1, bias = False).to(device)
        self.LinIn.weight, self.LinFeedback.weight = self.set_Win2()
        self.LinOut = nn.Linear(self.n_nodes +1, y.shape[1], bias = False).to(device)
        if not self.classification:
            self.out_activation = self.LinOut
        
        #build the state matrix:
        self.state = torch.zeros((x.shape[0], self.n_nodes), device = self.device).detach()
        self.state._name_ = "state"

        current_state = self.state[-1] 


        if not self.backprop:
            self.LinOut.requires_grad_, self.LinIn.requires_grad_, self.LinFeedback.requires_grad_ = False, False, False

        self.burn_in = burn_in

        # later add noise.
        self.inner_noise, self.outer_noise = False, False

        self.losses = None
            
        #fast exact solution ie we don't want to run backpropogation (ie we aren't doing classification):
        if not self.backprop:
            with torch.no_grad():

                #run through the states.
                for t in range(1, x.shape[0]):
                    self.state[t, :] = self.forward2(t, input_ = x[t].T, 
                                                        current_state = self.state[t-1,:], 
                                                        output_pattern = y[t-1]).squeeze()

                extended_states = torch.hstack((self.state, x))
                extended_states._name_ = "complete_data"

                train_x = extended_states[burn_in:, :]
                train_y = y[burn_in:]
                bias = None
                if not self.regularization:
                    print("no regularization")
                    pinv = torch.pinverse(train_x)
                    weight = torch.matmul(pinv,
                                          train_y )
                elif self.l2_prop == 1:
                    print("ridge regularizing")
                    
                    ridge_x = torch.matmul(train_x.T, train_x) + \
                                       self.regularization * torch.eye(train_x.shape[1], device = self.device)
                    ridge_y = torch.matmul(train_x.T, train_y)
                    ridge_x_inv = torch.pinverse(ridge_x)
                    weight = ridge_x_inv @ridge_y
                    assert 1 ==0
                    #torch.solve solves AX = B. Here X is beta_hat, A is ridge_x, and B is ridge_y
                    #weight = torch.solve(ridge_y, ridge_x).solution

                else:
                    #this needs more work, but it is in progress.
                    elastic_net_x = orig_X[burn_in:, :]
                    gram_matrix = torch.matmul(elastic_net_x.T, elastic_net_x) 

                    regr = ElasticNet(random_state=0, 
                                          alpha = self.regularization, 
                                          l1_ratio = 1-self.l2_prop,
                                          selection = "random",
                                          max_iter = 3000,
                                          tol = 1e-2,
                                          fit_intercept = True,
                                          precompute = gram_matrix.numpy()
                                          )
                    regr.fit(train_x.numpy(), train_y.numpy())

                    

                    weight = torch.tensor(regr.coef_, device = self.device)
                    bias = torch.tensor(regr.coef_, device =self.device)
                    """
                    try:
                    except:

                        print("elastic net regression Failing, falling back to ridge")
                        ridge_x = gram_matrix + \
                                       self.regularization * torch.eye(train_x.shape[1], device = self.device)
                        ridge_y = torch.matmul(train_x.T, train_y)
                        ridge_x_inv = torch.pinverse(ridge_x)
                        weight = ridge_x_inv @ridge_y
                    """

                    """
                    print("elastic net regularizing")
                    ridge_x = torch.matmul(train_x.T, train_x) + \
                                       self.regularization * torch.eye(train_x.shape[1], device = self.device)
                    ridge_y = torch.matmul(train_x.T, train_y)
                    ridge_x_inv = torch.pinverse(ridge_x)
                    weight = ridge_x_inv @ridge_y

                    with torch.enable_grad():
                        my_elastic = ElasticNetRegularization(iterations = 4000, 
                                          l2_proportion = self.l2_prop, 
                                          regularization_parameter = self.regularization,
                                          fail_tolerance = 30, val_prop = 0.9, learning_rate = self.reg_lr, ridge_weights = weight)
                        layer, losses = my_elastic.fit(train_x, train_y)
                        weight = layer.weight.detach()
                        bias = layer.bias.detach()
                        self.losses = losses
                    """

                #self.inverse_out_activation().T
                
                self.LinOut.weight = nn.Parameter(weight.view(-1,1), requires_grad = False)
                if type(bias) != type(None):
                    self.LinOut.bias = nn.Parameter(bias.view(-1,1), requires_grad = False)
                
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
            complete_data = torch.hstack((inputs, self.state))
        
        # Store last y value as starting value for predictions
        self.y_last = y[-1, :]

        # Return all data for computation or visualization purposes (Note: these are normalized)
        return extended_states, (y[1:,:] if self.feedback else y), burn_in
    
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
            #inputs = inputs.to(device)
            if keep:
                # Store for denormalization
                self._input_means = inputs.mean(axis=0)
                self._input_stds = inputs.std(dim = 0) #, ddof = 1)

            # Transform
            transformed.append((inputs - self._input_means) / self._input_stds)
            
            #self._input_means = self._input_means.to(device)
            #self._input_stds  = self._input_stds.to(device)

        if not outputs is None:
            #outputs = outputs.to(device)
            if keep:
                # Store for denormalization
                self._output_means = outputs.mean(axis=0)
                self._output_stds = outputs.std(dim = 0)#, ddof=1)

            # Transform
            transformed.append((outputs - self._output_means) / self._output_stds)
            
            self._output_means = self._output_means.to(device)
            self._output_stds = self._output_stds.to(device)
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
        y = y.to(device)
        
        # Run prediction
        final_t =y.shape[0]
        if steps_ahead is None:
            y_predicted = self.predict(n_steps = y.shape[0], x=x, y_start=y_start)
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            y_predicted = self.predict_stepwise(y, x, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]
        
        # Return error
        return self.error(y_predicted, y, scoring_method, alpha=alpha), y_predicted

    def test2(self, y, x=None, y_start=None, steps_ahead=None, scoring_method='nmse', alpha=1.):
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
        y = y.to(device)
        
        # Run prediction
        final_t =y.shape[0]
        if steps_ahead is None:
            y_predicted = self.predict2(n_steps = y.shape[0], x=x, y_start=y_start)
            #printc("predicting "  + str(y.shape[0]) + "steps", 'blue')
        else:
            y_predicted = self.predict_stepwise(y, x, steps_ahead=steps_ahead, y_start=y_start)[:final_t,:]
        
        # Return error
        return self.error(y_predicted, y, scoring_method, alpha=alpha), y_predicted
    
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
        # Check if ESN has been trained
        if  self.y_last is None: #self.out_weights is None or
            raise ValueError('Error: ESN not trained yet')
        
        # Normalize the inputs (like was done in train)
        if not self.already_normalized:
            if not x is None:
                x = self.normalize(inputs=x)

        #initialize input:
        inputs = Variable(torch.zeros((n_steps, 1)), requires_grad = False).to(device) #torch.ones((n_steps, 1), dtype=np.float32)  # Add bias term
        
        #Choose correct input
        if x is None and not self.feedback:
            #raise ValueError("Error: provide x or enable feedback")
            inputs = torch.ones((self.in_weights.shape[0], self.in_weights.shape[1])).to(device)
        elif x is not None:
            inputs = torch.hstack((inputs, x)).to(device)
        inputs._name_ = "inputs"
        
        # Set parameters
        if self.out_weights.shape[1] == 1:
            y_predicted = torch.zeros((n_steps,), dtype=torch.float32).to(device)
        else:
            y_predicted = torch.zeros((n_steps, self.out_weights.shape[1]), dtype=torch.float32).to(device)

        # Get last states
        previous_y = self.y_last
        
        #if not self.already_normalized:
        if not y_start is None: #if not x is None:
            previous_y = self.normalize(outputs=y_start)[0]

        # Initialize state from last availble in train
        current_state = self.state[-1]
        current_state._name_ = "current state"

        # Predict iteratively
        for t in range(n_steps):

            # Get correct input based on feedback setting
            current_input = inputs[t].T if not self.feedback else torch.hstack((inputs[t], previous_y)).to(device)

            # Update
            update = self.activation_function(torch.matmul(self.in_weights, current_input) + 
                                              torch.matmul(self.weights, current_state))
            #print("update: " + str(update.shape))
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state

            # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
            complete_row = torch.hstack((current_input, current_state))

            if self.out_weights.shape[1] > 1:
                y_predicted[t,:] = torch.matmul(complete_row, self.out_weights)
                previous_y = y_predicted[t,:]
            else:
                y_predicted[t] = torch.matmul(complete_row, self.out_weights)
                previous_y = y_predicted[t]

        # Denormalize predictions
        y_predicted = self.denormalize(outputs=y_predicted)
        return y_predicted.view(-1, self.out_weights.shape[1])

    def predict2(self, n_steps, pure_prediction = True, x=None, y_start=None, continuation = True):
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
        if self.y_last is None: #self.out_weights is None or 
            raise ValueError('Error: ESN not trained yet')
        
        # Normalize the inputs (like was done in train)
        if not x is None:
            x = self.normalize(inputs=x)
        
        # Set parameters
        if self.LinOut.weight.shape[0] == 1:
            y_predicted = torch.zeros((n_steps,), dtype=torch.float32).to(device)
        else:
            y_predicted = torch.zeros((n_steps, self.LinOut.weight.shape[0]), dtype=torch.float32).to(device)

        #if not self.already_normalized:
        n_samples = x.shape[0]

        if not y_start is None: #if not x is None:
            previous_y = self.normalize(outputs=y_start)[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_nodes)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = torch.vstack([lastinput, x]).view(-1,1)

        states = torch.zeros((n_samples + 1, self.n_nodes))
        states[0,:] = laststate

        outputs = torch.vstack(
            [lastoutput, torch.zeros((n_samples, self.n_outputs))])

        for t in range(n_samples):
            states[t + 1, :] = self.forward2(t, input_ = inputs[t + 1], 
                                                current_state = states[t, :], 
                                                output_pattern = outputs[t, :], 
                                                verbose = True)
            extended_state_spec = torch.cat([states[t+1,:], inputs[t+1, :]])
            #print("extended_state_spec", extended_state_spec.shape)
            #print("self.LinOut.weight", extended_state_spec.shape)
            outputs[t+1,:] = torch.dot(self.LinOut.weight.squeeze(), extended_state_spec.squeeze()) #torch.dot(self.LinOut.weight, 

        # Denormalize predictions
        #outputs = self.out_weights.T@outputs)#complete_data.T)#outputs=outputs[1:])
        #y_predicted.view(-1, self.n_outputs)
        #print("outputs", y_predicted.shape)
        return self.denormalize(outputs = outputs[1:]).view(-1, self.n_outputs) # 



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
            if not self.already_normalized:
                x = self.normalize(inputs=x)

        # Timesteps in y
        t_steps = y.shape[0]

        # Check input
        if not x is None and not x.shape[0] == t_steps:
            raise ValueError('x has the wrong size for prediction: x.shape[0] = {}, while y.shape[0] = {}'.format(
                x.shape[0], t_steps))

        # Choose correct input
        if x is None and not self.feedback:
            #pass #raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
            inputs = torch.ones((t_steps + steps_ahead, 2), dtype=torch.float32, device = self.device) 
        elif not x is None:
            # Initialize input
            inputs = torch.ones((t_steps, 1), dtype=torch.float32)  # Add bias term
            inputs = torch.hstack((inputs, x))  # Add x inputs
        else:
            # x is None
            inputs = torch.ones((t_steps + steps_ahead, 1), dtype=torch.float32, device = self.device)  # Add bias term
        
        # Run until we have no further inputs
        time_length = t_steps if x is None else t_steps - steps_ahead + 1

        # Set parameters
        y_predicted = torch.zeros((time_length, steps_ahead), dtype=torch.float32, device=self.device)

        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]#.to(device)

        # Initialize state from last availble in train
        current_state = self.state[-1]

        # Predict iteratively
        with torch.no_grad():
            
            for t in range(time_length):

                # State_buffer for steps ahead prediction
                prediction_state = current_state.clone().detach()
                
                # Y buffer for step ahead prediction
                prediction_y = previous_y.clone().detach()
            
                # Predict stepwise at from current time step
                for n in range(steps_ahead):
                    
                    # Get correct input based on feedback setting
                    prediction_input = inputs[t + n] if not self.feedback else torch.hstack((inputs[t + n], prediction_y))
                    
                    # Update
                    prediction_update = self.activation_function(torch.matmul(self.in_weights, prediction_input.T) + 
                                                   torch.matmul(self.weights, prediction_state))
                    
                    prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state
                    
                    # Store for next iteration of t (evolves true state)
                    if n == 0:
                        current_state = prediction_state.clone().detach()
                    
                    # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
                    prediction_row = torch.hstack((prediction_input, prediction_state))
                    if not self.backprop:
                        y_predicted[t, n] = torch.matmul(prediction_row, self.out_weights)
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
            loss = torch.zeros(1,1, device = self.device)
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
            error = torch.mean(torch.square(errors)) / torch.square(target.ravel().std())#ddof=1))
        elif method == 'nrmse':
            error = torch.sqrt(torch.mean(torch.square(errors))) / target.flatten().std()#ddof=1)
        elif method == 'tanh-nrmse':
            nrmse = torch.sqrt(torch.mean(torch.square(errors))) / target.ravel().std(ddof=1)
            error = alpha * torch.tanh(nrmse / alpha)
        elif method == 'log':
            mse = torch.mean(torch.square(errors))
            error = torch.log(mse)
        elif method == 'log-tanh':
            nrmse = torch.sqrt(torch.mean(torch.square(errors))) / target.ravel().std(ddof=1)
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
        
        if not inputs is None and not self.already_normalized:
            transformed.append((inputs * self._input_stds) + self._input_means)

        if not outputs is None:
            transformed.append((outputs * self._output_stds) + self._output_means)

        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]