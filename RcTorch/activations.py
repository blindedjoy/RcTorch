import torch
from torch.nn import Tanh

def _sech2(z):
    """
    Sech2 is the derivative of tanh.

    Parameters
    ----------
    z : pytorch.tensor
        tensor to perform the sech2 operation on

    Returns
    -------
    pytorch.tensor

    """
    return (1/(torch.cosh(z)))**2

def _sigmoid_derivative(z):
    """
    Derivative of the sigmoid function

    Parameters
    ----------
    z : pytorch.tensor
        tensor to perform the operation on

    Returns
    -------
    pytorch.tensor

    """
    s = torch.sigmoid(z)
    return s*(1-s)


def _nrmse(yhat,y):
    """
    Normalized root mean squared error loss function.

    Extended description of function.

    Parameters
    ----------
    yhat : torch.tensor
        the network prediction
    y : torch.tensor
        the ground truth data, that we would like the RC to fit

    Returns
    -------
    torch.tensor
        the error tensor

    """
    return torch.sqrt(torch.mean((yhat-y)**2)/torch.mean(y**2))

def _sinsq(x):
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
    return torch.square(torch.sin(x))

def _inverse_hyperbolic_tangent(z):
    """
    Inverse hyperbolic tangent function

    0.5 * log((1+z)/(1-z))

    Extended description of function.

    Parameters
    ----------
    z : pytorch.tensor
        Desc

    Returns
    -------
    pytorch.tensor

    """
    # z_max = z.abs().max()  + 0.0001
    # z = z/z_max
    return (1/2)*torch.log((1+z)/(1-z))

def _sech2_(z):
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
    return (1/(np.cosh(z)))**2

def _identity(z):
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
    return z

def _neg_sin(z):
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
    return -  torch.sin(z)

def _neg_double_sin(z):
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
    return - torch.sin(2 * z)


def _double_cos(z):
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
    return torch.cos(2 * z)

tanh_activation = Tanh()

tanh_at_2 = 0.9640275800
tanh_at_2_half = 0.48201379003

def _my_relu_i(z, lim = 2):
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
    if z >= lim:
        return np.tanh(z)#tanh_at_2
    elif z <= -lim:
        return np.tanh(z)#tanh_at_2 
    else:
        return tanh_at_2_half*z



def _rnn_relu(z):
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
    return z.apply_(_my_relu_i)


def _my_relu_i_prime(z, lim = 2):
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
    if (z >= lim) or (z <= -lim):
        return _sech2_(z)
    else:
        return tanh_at_2_half

def _rnn_relu_prime(z):
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
    return z.apply_(_my_relu_i_prime)


def _log_sin(z):
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
    return torch.sin(torch.log(z))

def _log_sin_prime(z):
    """
    sin(log(z))

    Parameters
    ----------
    z : torch.tensor
        input tensor

    Returns
    -------
    torch.tensor
        activated torch.tensor


    """
    return (1/x)*torch.cos(log(z))

def _sin2(z):
    """
    sin squared

    Parameters
    ----------
    z : torch.tensor
        input tensor

    Returns
    -------
    torch.tensor
        activated torch.tensor

    """
    s = torch.sin(5 * z)*torch.sin(5*z)*2 - 1
    return s**2

def _sin2_derivative(z):
    """
    derivative of sin2

    Parameters
    ----------
    z : torch.tensor
        input tensor

    Returns
    -------
    torch.tensor
        activated torch.tensor

    """
    s = 10*torch.sin(10 * z)
    return s**2

def _sincos(z):
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
    return torch.sin(z)*torch.cos(z)

def _sincos_derivative(z):
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
    return torch.cos(2*z)


def _convert_activation_f(string, derivative  = False, both = True):
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

    if string == "sigmoid":
        act_f, act_f_prime =  torch.sigmoid, _sigmoid_derivative
    elif string == "tanh":
        act_f, act_f_prime =   torch.tanh, _sech2
    elif string == "sin":
        act_f, act_f_prime =   torch.sin, torch.cos
    elif string == "cos":
        act_f, act_f_prime =   torch.cos, _neg_sin
    elif string == "double_cos":
        act_f, act_f_prime =   _double_cos, _neg_double_sin
    elif string == "relu":
        act_f, act_f_prime =   _rnn_relu, _rnn_relu_prime
    elif string == "log_sin":
        act_f, act_f_prime =   _log_sin, _log_sin_prime
    elif string == "sin2":
        act_f, act_f_prime =   _sin2, _sin2_derivative
    elif string == "sincos":
        act_f, act_f_prime =   _sincos, _sincos_derivative
    else:
        assert False, f"activation function '{activation_function}' not yet implimented"
    if both:
        return act_f, act_f_prime
    if not derivative:
        return act_f
    else:
        return act_f_prime
