
import torch
import torch.nn as nn
from torch.nn import KLDivLoss

class custom_kl_loss(nn.Module):
    """
    An example custom loss function
    
    See this link for more information:
    https://pdf.co/blog/deep-learning-pytorch-custom-loss-function
    
    Parameters
    ----------
        temp: The temperature argument for softmax (will soften the probability distribution)
    
    Returns
    ----------
        penalized loss metric
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.temp = kwargs.get('temp', 1)
        
    def Tsoftmax(self, x, log = True):
        """ Temperature softens the output empirical distribution
        #https://stackoverflow.com/questions/44081007/logsoftmax-stability
        
        Parameters
        ----------
        x: torch.FloatTensor
            the vector to apply softmax to
        
        log: boolean
            if True returns the log softmax which is more stable because it lacks division
            otherwise return the normal softmax.
            
        #log(softmax) = x - b - log(x-b)exp(b)
        
        If we set formula, this new equation has both overflow and underflow stability conditions.
        You can also check this link for a little more descriptions:
        
        https://stackoverflow.com/questions/44081007/logsoftmax-stability
        """
        if log:
            #b = max(x)/self.temp
            #The torch.max function returns a tuple: input_max, input_indexes
            
            b  = torch.max(x)
            
            first_term = (x - b)/self.temp
            second_term = torch.log(torch.sum(torch.exp(first_term)))
            softmax = first_term - second_term
        else:
            exp_x = torch.exp(x/self.temp)
            softmax = exp_x / torch.sum(exp_x)
        return softmax

    def soften(self, target, input_, ):
        """
        kl expectes log inputs and probability outputs,
        From the documentation:
        
        As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
        *log-probabilities* and is not restricted to a 2D Tensor.
        The targets are interpreted as *probabilities* by default, but could be considered
        as *log-probabilities* with :attr:`log_target` set to ``True``.

        """
        soft_input = self.Tsoftmax(input_.view(-1,), log = True)
        
        soft_target = self.Tsoftmax(target.view(-1,), log = True)
        return soft_target, soft_input

    def forward(self, target, predicted, _lambda = 1, upper_error_limit = 100000, min_kl = None,  **kwargs):
        """
        Parameters:
            target: The target sequenc3
            prediction: The prediction sequence
            _lambda: a hyper parameter controlling the kl_penalty

        """
        #assert False, f'target {target.shape} predicted {predicted.shape}'
        self.kl =KLDivLoss(log_target = True)

        self.target = target
        self.predicted =  predicted
        self.errors = self.target - self.predicted

        soft_target, soft_input = self.soften(target, predicted)
        
        min_kl = self.kl(soft_target, soft_input)
        
        loss2 = torch.exp(self.kl(soft_target, soft_input)) - min_kl
        
        loss1 =  torch.mean(torch.square(self.errors))/torch.square(self.target).mean()

        assert loss1 >= 0, loss1
        assert loss2 >= 0, loss1
        
        
        loss = loss1 + _lambda * loss2

        #print(f'Loss: {loss1 } {loss2} {loss}')
        
        return loss

def odeLoss(time, N, dN_dx):
    f = 1 - torch.exp(-time)
    loss = f * dN_dx + N - 1
    loss = torch.square(loss).mean()
    return loss


