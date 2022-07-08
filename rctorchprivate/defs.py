import torch
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt
import torch.optim as optim
from decimal import Decimal
import numpy as np
import ray

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

lam =1
def hamiltonian(x, p, lam = lam):
    return (1/2)*(x**2 + p**2) + lam*x**4/4

def hamiltonian2(x, p, lam = lam):
    return 1.2 + (1/2)*p**2 -torch.cos(x)- (1 / 5) * torch.cos(5*x)

def fforce(t, A = 1):
    return A * torch.sin(t)

def no_fforce(t):
    return 0

def no_reg_loss(X , y, ydot, out_weights, force_t = None, 
                reg = True, ode_coefs = None, mean = True,
               enet_strength = None, enet_alpha = None, init_conds = None, lam = 1):
    """

    Arguments
    ---------
    X: dtype
        Desc
    y: dtype
        Desc
    out_weight : dtype


    Returns
    -------

    L: dtype
        desc


    """
    y, p = y[:,0].view(-1,1), y[:,1].view(-1,1)
    ydot, pdot = ydot[:,0].view(-1,1), ydot[:,1].view(-1,1)
    
    #with paramization
    L =  (ydot - p)**2 + (pdot + y + lam * y**3   - force_t)**2
    
    #if mean:
    L = torch.mean(L)
    #    hi = float(L)
    
    #if reg:
    weight_size_sq = torch.mean(torch.square(out_weights))
    weight_size_L1 = torch.mean(torch.abs(out_weights))
    L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    L = L + 0.1 * L_reg 
    

#     y0, p0 = init_conds
#     ham = hamiltonian(y, p)
#     ham0 = hamiltonian(y0, p0)
#     L_H = (( ham - ham0).pow(2)).mean()
#     assert L_H >0

#     L = L +  L_H
    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L

def H(x, y, px, py):
    xsq, ysq = x**2, y**2
    LHt1 = 0.5*(px**2 + py**2)
    LHt2 = 0.5*(xsq + ysq)
    LHt3 = xsq*y - (y**3)/3
    LH = LHt1 + LHt2 + LHt3
    return(LH)

# def force(X, A = 0):
#     return torch.zeros_like(X)

def hennon_hailes_loss(X , y, ydot, out_weights, force_t = None, 
                        reg = True, ode_coefs = None, mean = True, custom_loss = None,
                        enet_strength = None, enet_alpha = None, init_conds = None, lam = 1):
    

    #4 outputs
    assert init_conds

    x, y, px, py = [y[:,i].view(-1,1) for i in range(y.shape[1])]#, y[:,1].view(-1,1)
    dx, dy, dpx, dpy = [ydot[:,i].view(-1,1)for i in range(ydot.shape[1])]

    x0, y0, px0, py0 = init_conds

    xsq, ysq = x**2, y**2
    
    #with paramization
    term1 = (dx - px)**2 
    term2 = (dy - py)**2
    term3 = (dpx + x + 2*x*y)**2 
    term4 = (dpy + y + xsq - ysq )**2
    L = (term1 + term2 + term3 + term4).mean()

    
    #hamiltonian regularization
    LH_ = H(x, y, px, py)
    LH0 = H(x0, y0, px0, py0)
    L = L + (LH_ -LH0).pow(2).mean()


    #if reg:
    weight_size_sq = torch.mean(torch.square(out_weights))
    weight_size_L1 = torch.mean(torch.abs(out_weights))
    L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    L = L + 0.1 * L_reg 
    
    return L

def elastic_loss(X , y, ydot, out_weights, force_t = fforce, 
                reg = True, ode_coefs = None, mean = True,
               enet_strength = None, enet_alpha = None, init_conds = None, lam = 1):
    y, p = y[:,0].view(-1,1), y[:,1].view(-1,1)
    ydot, pdot = ydot[:,0].view(-1,1), ydot[:,1].view(-1,1)
    #with paramization
    L =  (ydot - p)**2 + (pdot + y + lam * y**3   - force_t)**2
    
    #if mean:
    L = torch.mean(L)
    #    hi = float(L)
    
    #if reg:
    weight_size_sq = torch.mean(torch.square(out_weights))
    weight_size_L1 = torch.mean(torch.abs(out_weights))
    L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    L = L + 0.1 * L_reg 
    

#     y0, p0 = init_conds
#     ham = hamiltonian(y, p)
#     ham0 = hamiltonian(y0, p0)
#     L_H = (( ham - ham0).pow(2)).mean()
#     assert L_H >0

#     L = L +  L_H
    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L

def freparam(t, order = 1):
    exp_t = torch.exp(-t)
    
    derivatives_of_g = []
    
    g = 1 - exp_t
    
    #0th derivative
    derivatives_of_g.append(g)
    
    g_dot = 1 - g
    return g, g_dot

def driven_pop_loss(X , y, ydot, out_weights, lam = 1, force_t = None, reg = False, 
               ode_coefs = None, init_conds = None, 
                enet_alpha = None, enet_strength =None, mean = True):
    
    #with paramization
    L =  ydot  + lam * y - force_t
    
    # if reg:
    #     #assert False
    #     weight_size_sq = torch.mean(torch.square(out_weights))
    #     weight_size_L1 = torch.mean(torch.abs(out_weights))
    #     L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    #     L = L + 0.1 * L_reg 
    
    L = torch.square(L)
    if mean:
        L = torch.mean(L)
    return L

def ham_loss(X , y, ydot, out_weights, force_t = None, 
                reg = True, ode_coefs = None, mean = True,
               enet_strength = None, enet_alpha = None, init_conds = None, lam = 1):
    y, p = y[:,0].view(-1,1), y[:,1].view(-1,1)
    ydot, pdot = ydot[:,0].view(-1,1), ydot[:,1].view(-1,1)
    
    #with paramization
    L =  (ydot - p)**2 + (pdot + y + lam * y**3   - force_t).pow(2)
    
    #if mean:
    L = torch.mean(L)
    
    # #if reg:
    # weight_size_sq = torch.mean(torch.square(out_weights))
    # weight_size_L1 = torch.mean(torch.abs(out_weights))
    # L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    # L = L + 0.1 * L_reg 
    

    y0, p0 = init_conds
    ham = hamiltonian(y, p)
    ham0 = hamiltonian(y0, p0)
    L_H = (( ham - ham0).pow(2)).mean()
    assert L_H >0

    L = L +  L_H
    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L

def multi_attractor_loss(X , y, ydot, out_weights, force_t = None, 
                reg = True, ode_coefs = None, mean = True,
               enet_strength = None, enet_alpha = None, init_conds = None, lam = 1):
    y_, p = y[:,0].view(-1,1), y[:,1].view(-1,1)

    ydot, pdot = ydot[:,0].view(-1,1), ydot[:,1].view(-1,1)

    #with paramization
    L =  (ydot - p)**2 + (pdot + torch.sin(y_) + torch.sin(5*y_) )**2 
    #+(ydot - ydot_hat)**2 + (pdot - pdot_hat**2)
    
    #if mean:
    L = torch.mean(L)
    
    if reg:
        #assert False
        weight_size_sq = torch.mean(torch.square(out_weights))
        weight_size_L1 = torch.mean(torch.abs(out_weights))
        L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
        L = L + 0.1 * L_reg 

    y0, p0 = init_conds
    ham = hamiltonian(y, p)
    ham0 = 1.2 + (1/2)*p0**2 -np.cos(y0)-(1/5)*np.cos(5*y0)
    L_H = (( ham - ham0).pow(2)).mean()
    assert L_H >0

    L = L +  0.1 * L_H
    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L

# global lr_100_epoch
# global lr_1000_epoch
# global lr_5000_epoch

# lr_100_epoch, lr_1000_epoch, lr_5000_epoch = [], [], []   


def optimize_last_layer(esn, 
                        SAVE_AFTER_EPOCHS = 1,
                        epochs = 45000,
                        custom_loss = None,
                        EPOCHS_TO_TERMINATION = None,
                        f = None,
                        lr = 0.05, 
                        reg = None,
                        plott = False,
                        plot_every_n_epochs = 100,
                        force_t = None):#gamma 0.1, spikethreshold 0.07 works
    with torch.enable_grad():
        #define new_x
        new_X = esn.extended_states.detach()
        spikethreshold = esn.spikethreshold

        #force detach states_dot
        esn.states_dot = esn.states_dot.detach().requires_grad_(False)

        #define criterion
        criterion = torch.nn.MSELoss()

        #assert esn.LinOut.weight.requires_grad and esn.LinOut.bias.requires_grad
        #assert not new_X.requires_grad

        #define previous_loss (could be used to do a convergence stop)
        previous_loss = 0

        #define best score so that we can save the best weights
        best_score = 0

        #define the optimizer
        optimizer = optim.Adam(esn.parameters(), lr = lr)

        #optimizer = torch.optim.SGD(model.parameters(), lr=100)
        if esn.gamma_cyclic:
            cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 10**-6, 0.01,
                                            gamma = esn.gamma_cyclic,#0.9999,
                                            mode = "exp_range", cycle_momentum = False)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=esn.gamma)
        lrs = []

        #define the loss history
        loss_history = []

        if plott:
          #use pl for live plotting
          fig, ax = pl.subplots(1,3, figsize = (16,4))

        t = esn.X#.view(*N.shape).detach()
        #force_t = force(t)
        g, g_dot = esn.G
        y0  = esn.init_conds[0]

        flipped = False
        flipped2 = False
        pow_ = -4
        floss_last = 0


        try:
            assert esn.LinOut.weight.requires_grad and esn.LinOut.bias.requires_grad
        except:
            esn.LinOut.weight.requires_grad_(True)
            esn.LinOut.bias.requires_grad_(True)

        y0, p0 = esn.init_conds

        #ham0 = (1/2)*p0**2 - 3*y0**2 + (21/4)*y0**4
        #bail

        #begin optimization loop
        for e in range(epochs):

            optimizer.zero_grad()

            N = esn.forward( esn.extended_states )
            N_dot = esn.calc_Ndot(esn.states_dot)

            y = g *N 

            ydot = g_dot * N + g * N_dot
            
            for i in range(y.shape[1]):
                
                y[:,i] = y[:,i] + esn.init_conds[i]
                
            #y[:,1] = y[:,1] + esn.init_conds[1]

            #assert N.shape == N_dot.shape, f'{N.shape} != {N_dot.shape}'

            #assert esn.LinOut.weight.requires_grad and esn.LinOut.bias.requires_grad

            #total_ws = esn.LinOut.weight.shape[0] + 1
            #weight_size_sq = torch.mean(torch.square(esn.LinOut.weight))

            loss = custom_loss(esn.X, y, ydot, esn.LinOut.weight, reg = reg, ode_coefs = esn.ode_coefs, init_conds = esn.init_conds, 
                    enet_alpha= esn.enet_alpha, enet_strength = esn.enet_strength, force_t = force_t)
            loss.backward()
            optimizer.step()
            if esn.gamma_cyclic and e > 100 and e <5000:
                cyclic_scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])

            loss_history.append(loss)

            # if e == 10**3:
            #     if floss > 10**(5):
            #         EPOCHS_TO_TERMINATION = e + 50

            # if e == 10**4:
            #     if floss > 10**(2.5):
            #         EPOCHS_TO_TERMINATION = e + 50
                    
            if e > 0:
                loss_delta = float(torch.log(floss_last) - torch.log(loss)) 
                if loss_delta > esn.spikethreshold:# or loss_delta < -3:
                    lrs.append(optimizer.param_groups[0]["lr"])
                    scheduler.step()


            # if not e and not best_score:
            #     best_bias, best_weight, best_fit = esn.LinOut.bias.detach(), esn.LinOut.weight.detach(), y.clone()

            if e > SAVE_AFTER_EPOCHS:
                if not best_score:
                    best_score = min(loss_history)
                if loss < best_score:  
                    best_bias, best_weight = esn.LinOut.bias.detach(), esn.LinOut.weight.detach()
                    best_score = loss
                    best_fit = y.clone()
                    best_ydot = ydot.clone()
            floss_last = loss
            # else:
            #     if floss < best_score:
            #         best_bias, best_weight = esn.LinOut.bias.detach(), esn.LinOut.weight.detach()
            #         best_score = float(loss)
            #         best_fit = y.clone()
            #         best_ydot = ydot.clone()
            
            # if e >= EPOCHS_TO_TERMINATION and EPOCHS_TO_TERMINATION:
            #     return {"weights": best_weight, "bias" : best_bias, "y" : best_fit, 
            #           "loss" : {"loss_history" : loss_history},  "best_score" : torch.tensor(best_score),
            #           "RC" : esn}
            
            # if plott and e:

            #     if e % plot_every_n_epochs == 0:
            #         for param_group in optimizer.param_groups:
            #             print('lr', param_group['lr'])
            #         ax[0].clear()
            #         logloss_str = 'Log(L) ' + '%.2E' % Decimal((loss).item())
            #         delta_loss  = ' delta Log(L) ' + '%.2E' % Decimal((loss-previous_loss).item())

            #         print(logloss_str + ", " + delta_loss)
            #         ax[0].plot(y.detach().cpu())
            #         ax[0].set_title(f"Epoch {e}" + ", " + logloss_str)
            #         ax[0].set_xlabel("t")

            #         ax[1].set_title(delta_loss)
            #         ax[1].plot(ydot.detach().cpu(), label = "ydot")
            #         #ax[0].plot(y_dot.detach(), label = "dy_dx")
            #         ax[2].clear()
            #         #weight_size = str(weight_size_sq.detach().item())
            #         #ax[2].set_title("loss history \n and "+ weight_size)

            #         ax[2].loglog(loss_history)
            #         ax[2].set_xlabel("t")

            #         #[ax[i].legend() for i in range(3)]
            #         previous_loss = loss.item()

            #         #clear the plot outputt and then re-plot
            #         display.clear_output(wait=True) 
            #         display.display(pl.gcf())


        return {"weights": best_weight, "bias" : best_bias, "y" : best_fit, "ydot" : best_ydot, 
              "loss" : {"loss_history" : loss_history}, "best_score" : best_score,
              "RC" : esn}

def hamiltonian3(x, p, lam = lam):
    return (1/2)*p.pow(2) - 3*x.pow(2) + (21/4)*x.pow(4)

def dual_loss(X , y, ydot, out_weights, force_t = None, 
                reg = True, ode_coefs = None, mean = True,
               enet_strength = None, enet_alpha = None, init_conds = None, lam = 1, ham0 = None):
    y, p = y[:,0], y[:,1]
    ydot, pdot = ydot[:,0], ydot[:,1]
    
    #with paramization
    L =  (ydot - p).pow(2) + (pdot - 6*y + 21*y.pow(3) ).pow(2)
    
    if reg:
        #assert False
        weight_size_sq = torch.mean(torch.square(out_weights))
        weight_size_L1 = torch.mean(torch.abs(out_weights))
        L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
        L = L + 0.1 * L_reg 

    #y0, p0 = init_conds
    ham = hamiltonian3(y, p)

    L_H = ( ham - ham0).pow(2)

    L = L + L_H

    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L.mean()

CUDAA = torch.cuda.is_available()
if CUDAA:
    #print("cuda is available")
    n_gpus = 0.1
else:
    #print("cuda is not available")
    n_gpus = 0

 
#weight_dict = backprop_f(self, force_t = self.force_t, custom_loss = ODE_criterion, epochs = epochs)

