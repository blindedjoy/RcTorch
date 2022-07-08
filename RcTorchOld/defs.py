import torch
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt
import torch.optim as optim
from decimal import Decimal
import numpy as np

lam =1
def hamiltonian(x, p, lam = lam):
    return (1/2)*(x**2 + p**2) + lam*x**4/4


def fforce(t, A = 1):
    return A * torch.sin(t)

def no_fforce(t):
    return 0

def no_reg_loss(X , y, ydot, out_weights, force_t = None, 
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

def H(x, y, px, py):
    xsq, ysq = x**2, y**2
    LHt1 = 0.5*(px**2 + py**2)
    LHt2 = 0.5*(xsq + ysq)
    LHt3 = xsq*y - (y**3)/3
    LH = LHt1 + LHt2 + LHt3
    return(LH)

def hennon_hailes_loss(X , y, ydot, out_weights, force_t = fforce, 
                reg = True, ode_coefs = None, mean = True,
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
    L = (term1 + term2 + term3 + term4).mean().pow(2)

    
    #hamiltonian regularization
    LH_ = H(x, y, px, py)
    LH0 = H(x0, y0, px0, py0)
    L = L + ((LH_ -LH0).mean()).pow(2)


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
    L =  (ydot - p)**2 + (pdot + y + lam * y**3   - force_t)**2
    
    #if mean:
    L = torch.mean(L)
    #    hi = float(L)
    
    #if reg:
    weight_size_sq = torch.mean(torch.square(out_weights))
    weight_size_L1 = torch.mean(torch.abs(out_weights))
    L_reg = enet_strength*(enet_alpha * weight_size_sq + (1- enet_alpha) * weight_size_L1)
    L = L + 0.1 * L_reg 
    

    y0, p0 = init_conds
    ham = hamiltonian(y, p)
    ham0 = hamiltonian(y0, p0)
    L_H = (( ham - ham0).pow(2)).mean()
    assert L_H >0

    L = L +  L_H
    
    #print("L1", hi, "L_elastic", L_reg, "L_H", L_H)
    return L

global lr_100_epoch
global lr_1000_epoch
global lr_5000_epoch

lr_100_epoch, lr_1000_epoch, lr_5000_epoch = [], [], []   

def optimize_last_layer(esn, 
                        SAVE_AFTER_EPOCHS = 1000,
                        epochs = 1000,#45000,
                        custom_loss = None,
                        loss_threshold = 10**-10,#10 ** -8,
                        EPOCHS_TO_TERMINATION = None,
                        f = fforce,
                        lr = 0.05, 
                        reg = None,
                        plott = False,
                        plot_every_n_epochs = 2499):#gamma 0.1, spikethreshold 0.07 works
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
        
    #bail
    
    #begin optimization loop
    for e in range(epochs):

        optimizer.zero_grad()
        
        N = esn.forward( esn.extended_states )
        N_dot = esn.calc_Ndot(esn.states_dot)

        y = g *N 
        
        ydot = g_dot * N + g * N_dot
        
        y[:,0] = y[:,0] + esn.init_conds[0]
        y[:,1] = y[:,1] + esn.init_conds[1]

        assert N.shape == N_dot.shape, f'{N.shape} != {N_dot.shape}'
        
        #assert esn.LinOut.weight.requires_grad and esn.LinOut.bias.requires_grad

        #total_ws = esn.LinOut.weight.shape[0] + 1
        #weight_size_sq = torch.mean(torch.square(esn.LinOut.weight))
        
        loss = custom_loss(esn.X, y, ydot, esn.LinOut.weight, reg = reg, ode_coefs = esn.ode_coefs,
                          init_conds = esn.init_conds, enet_alpha= esn.enet_alpha, enet_strength = esn.enet_strength)
        loss.backward()
        optimizer.step()
        if esn.gamma_cyclic and e > 100 and e < 4000:
            cyclic_scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        
        
        floss = float(loss)
        loss_history.append(floss)
        
        
        """
        if floss < 10**pow_ :
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            #for param_group in optimizer.param_groups:
            #    print('lr', param_group['lr'])
            pow_ -= 1
        """
#         if e == 10**3:
#             if floss > 10**(5):
#                 EPOCHS_TO_TERMINATION = e + 50
        
#         if e == 10**4:
#             if floss > 10**(2.5):
#                 EPOCHS_TO_TERMINATION = e + 50
        
        if e > 0:
            
            loss_delta = float(np.log(floss_last) - np.log(floss)) 
            if loss_delta > esn.spikethreshold:# or loss_delta < -3:
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler.step()
        
                
        quant = 0.8
        if e == 100:
            lr_100_epoch.append(floss)
            if floss >= np.quantile(np.array(lr_100_epoch), 0.8):
                if len(lr_100_epoch) > 20:
                    EPOCHS_TO_TERMINATION = e + 50
        elif e == 1000:
            lr_1000_epoch.append(floss)
            if floss >= np.quantile(np.array(lr_1000_epoch), 0.8):
                if len(lr_100_epoch) > 20:
                    EPOCHS_TO_TERMINATION = e + 50
        if e == 5000:
            lr_5000_epoch.append(floss)
            if floss >= np.quantile(np.array(lr_5000_epoch), 0.8):
                if len(lr_100_epoch) > 20:
                    EPOCHS_TO_TERMINATION = e + 50
        
        
        if not e and not best_score:
            best_bias, best_weight, best_fit = esn.LinOut.bias.detach(), esn.LinOut.weight.detach(), y.clone()

        if e > SAVE_AFTER_EPOCHS:
            if not best_score:
                
                if floss <= min(loss_history):
                    best_fit = y.clone()
                    best_ydot = ydot.clone()
                    best_bias, best_weight = esn.LinOut.bias.detach(), esn.LinOut.weight.detach()
                    best_score = float(loss)
            else:
                if floss < best_score:
                    best_fit = y.clone()
                    best_ydot = ydot.clone()
                    best_bias, best_weight = esn.LinOut.bias.detach(), esn.LinOut.weight.detach()
                    best_score = float(loss)
                    
        if not EPOCHS_TO_TERMINATION:
            if float(loss) < loss_threshold:
                EPOCHS_TO_TERMINATION = e + 100
        else:
            if e >= EPOCHS_TO_TERMINATION:
                return {"weights": best_weight, "bias" : best_bias, "fit" : best_fit, 
                        "loss" : loss_history, "best_score" : torch.tensor(best_score),
                        "RC" : esn}
        floss_last = floss
        if plott and e:

            if e % plot_every_n_epochs == 0:
                for param_group in optimizer.param_groups:
                    print('lr', param_group['lr'])
                ax[0].clear()
                logloss_str = 'Log(L) ' + '%.2E' % Decimal((loss).item())
                delta_loss  = ' delta Log(L) ' + '%.2E' % Decimal((loss-previous_loss).item())

                print(logloss_str + ", " + delta_loss)
                ax[0].plot(y.detach().cpu(), label = "exact")
                ax[0].set_title(f"Epoch {e}" + ", " + logloss_str)
                ax[0].set_xlabel("t")

                ax[1].set_title(delta_loss)
                ax[1].plot(N_dot.detach().cpu())
                #ax[0].plot(y_dot.detach(), label = "dy_dx")
                ax[2].clear()
                #weight_size = str(weight_size_sq.detach().item())
                #ax[2].set_title("loss history \n and "+ weight_size)

                ax[2].loglog(loss_history)
                ax[2].set_xlabel("t")

                [ax[i].legend() for i in range(3)]
                previous_loss = loss.item()

                #clear the plot outputt and then re-plot
                display.clear_output(wait=True) 
                display.display(pl.gcf())
            
            
    return {"weights": best_weight, "bias" : best_bias, "fit" : best_fit, 
            "loss" : loss_history, "best_score" : torch.tensor(best_score),
            "RC" : esn}

