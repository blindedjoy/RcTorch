import torch

def _dfx(inputs, outputs, retain_graph = True, create_graph = True):
    """
    runs torch.grad to find the gradient of outputs/inputs

    for more information see: https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch-autograd-grad

    Parameters
    ----------
    inputs  : [tensor]
        the inputs
    outputs : [tensor]
        the outputs
    retain_graph : bool
        same as in torch.autograd.grad
    grad_outputs : dtype
        same as torch.autograd.grad

    Returns
    -------
    torch.tensor
        the gradient of (outputs wrt inputs)

    """
    return torch.autograd.grad([outputs],[inputs], grad_outputs=torch.ones_like(f), create_graph = create_graph, retain_graph = retain_graph)[0]
