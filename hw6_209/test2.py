import numpy as np
from numpy import loadtxt

from matplotlib import pyplot as plt

# This is a class for reservoir computing
# The pyESN.py file must be in the same directory with this notebook
#from pyESN import ESN
from reservoir import *
import torch
import numpy as np
import time
import pylab as pl
from IPython import display


#RUN THIS CELL 
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
import os
import pathlib
working_dir = pathlib.Path().absolute()
# Uncomment the line below to help debug if the path to included images don't show
#print(working_dir)
os.chdir(working_dir)

dataX = np.loadtxt('./data/x.dat')
dataY = np.loadtxt('./data/y.dat')
dataZ = np.loadtxt('./data/z.dat')

def torchify(x):
    return torch.tensor(x).type(torch.float32).view(-1,1)

dataX, dataY, dataZ = [torchify(x) for x in (dataX, dataY, dataZ)]

# HELPER FUNCTIONS GO HERE

def myMSE(prediction,target):
    return np.sqrt(np.mean((prediction.flatten() - target.flatten() )**2))

def residuals(prediction,target):
    return (target.flatten() - prediction.flatten())

    
def prepareData(target, train_perc=0.9, plotshow=False):
    datalen =  len(target)        
    trainlen = int(train_perc*datalen)
    testlen  = datalen-trainlen

# Train/Test sets
    trainTarget = target[:trainlen]
    testTarget  = target[trainlen:trainlen+testlen]    
    inputTrain = np.ones(trainlen)
    inputTest  = np.ones(testlen)
        
    if plotshow:
        plt.figure(figsize=(14,3))
        plt.plot(range(0,trainlen), trainTarget,'g',label='Train')
        plt.plot(range(trainlen,trainlen+testlen), testTarget,'-r',label='Test')
        plt.legend(loc=(0.1,1.1),fontsize=18,ncol=2)
        plt.tight_layout()
        
    return trainTarget, testTarget, inputTrain, inputTest

def prepare_data(data, train_split):
    split_idx = int(len(data) * train_split)
    
    return_dict = {}
    
    return_dict['train']= data[:split_idx]
    return_dict['target'] = data[split_idx:]
    return return_dict

def plot_train_prediction(RC_, training_set):
    output_unnorm = RC.LinOut.weight.T@vals[0].T + RC.LinOut.bias
    hi = RC_._output_stds* (output_unnorm)+ RC._output_means
    hi = hi.view(-1,).numpy()

    plt.figure( figsize = (16,5))
    plt.title( "RC — training set prediction")
    plt.plot( output_unnorm.view(-1,), color = "yellow")
    plt.plot( training_set, linewidth =4, color = "cyan", label = "ground truth")
    plt.plot( training_set, linewidth =2, color = "blue", alpha = 0.9)
    plt.plot( hi, '--', color = "red", label = "train prediction")
    plt.xlabel("timestep")
    plt.ylabel("y")

    plt.legend();

def plot_states(RC_):
    plt.figure(figsize = (16,4))
    for i in range(4):
        plt.plot(RC_.state.T[-(i+1), :], alpha = 0.6);
    plt.title("RC Example States")
    plt.ylabel("y")
    plt.xlabel("timestep")

def plot_test_prediction(test_set, pred_):
    #plt.plot(test_outputs.T[-1,:100])
    plt.figure(figsize = (16,4))
    plt.plot(test_set, label = "ground truth");
    plt.plot(pred_, label = "prediction")
    plt.title("unoptimized hyper-parameter test set prediction")
    plt.legend()
    plt.show()

X95 = prepare_data(dataX, 0.95)
Y95 = prepare_data(dataY, 0.95)



### your code here
#%%time
#### eps isn't what it used to be.

bounds_dict = {"spectral_radius" : (1,2), 
               "connectivity" : (-4,0.5), 
               "regularization": (-5,3),
               "leaking_rate" : (0, 1),
               #"feedback_scaling" : (0,1),
               "l2_prop" : 1.0,
               "bias" : (-0.5, 0.5),
               #"noise" : (-5,-1),#"uniform",#
               "PyESNnoise":0.0,
               "n_nodes" : 1000, 
               "feedback": 1}
esn_cv = EchoStateNetworkCV(bounds = bounds_dict, subsequence_length = 7000, esn_feedback = True,
                            steps_ahead = None, scoring_method = "nmse", interactive = True, random_seed = 123,
                            initial_samples = 10, approximate_reservoir = False, length_min = 2**-9,
                            batch_size = 4, backprop = False, esn_burn_in = 0, validate_fraction = 0.2,
                            activation_function = nn.Tanh(), cv_samples = 2
                            ) #activation_function = sinsq,

results_round1 = esn_cv.optimize(y = X95['train'])

pl.plot(esn_cv.iteration_durations)
print(np.mean(esn_cv.iteration_durations))
results_round1;