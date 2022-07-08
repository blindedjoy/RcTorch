import torch

def _printn(param: torch.nn.parameter):
    """
    Print torch.nn.parameter

    Parameters
    ----------
    param torch.nn.parameter

    Returns
    -------
    int
        Description of return value
    """
    print(param._name_ + "\t \t", param.shape)