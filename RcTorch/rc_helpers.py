import re
import torch
import numpy as np

def _check_x(X, y, tensor_args = {}, supervised = False):
    """
    Make sure X is a valid input. 
    X is typically an observer, an input time series for a parameter aware RC

    Parameters
    ----------
    X : torch.tensor
        Observer time series
    y : torch.tensor
        target time series
    tensor_args : dict
        arguments to be fed to X, for example device and dtype
    supervised : bool
        supervised training or not (unsupervised data-less ODE solution)

    Returns
    -------
    X: torch.tensor
        valid X input (2d, on device etc)
    """
    if X is None:
        if supervised:
            X = torch.ones((y.shape[0],1), **tensor_args)
        else:
            X = torch.linspace(0, 1, steps = y.shape[0], **tensor_args)
    elif type(X) == np.ndarray:
        X = torch.tensor(X,  **tensor_args)
    
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    return X

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

def _printc(string_, color_, end = '\n') :
    """
    Print colored 

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

def _convert_ode_coefs(ode_coefs_, X):
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
                assert False, "ode_coefs must be a list of floats or strings of the form 't^pow', where pow is a real number., coef: {coef}"
    else:
        assert False, "ode_coefs must be a list of floats or strings of the form 't^pow', where pow is a real number., coef: {coef}"
    return ode_coefs