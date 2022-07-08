import ray
import torch

from .custom_loss import *
from .defs import *

@ray.remote(num_gpus=n_gpus, max_calls=1)
def execute_backprop(args,  
                     y0, 
                     ):
    """
    Parallelized backpropagation
    
    Parameters
    ----------
    args : dtype
        Desc
    y0 : dtype
        Description of arg1
    lr : float
        learning rate
    plott : bool
        if True ...
    reg : dtype
        Regularization for the loss function
    plot_every_n_epochs : int
        plotting interval
    SAVE_AFTER_EPOCHS : int
        begin saving the best weights after this many epochs

    Returns
    -------
    dict
        {"weights": best_weight, "bias" : best_bias, "y" : best_fit, "ydot" : best_ydot, 
          "loss" : {"loss_history" : loss_history}, "best_score" : torch.tensor(best_score)}
    """

    #old default args
    # epochs = 45000,
    # f = force,
    # lr = 0.05, 
    # plott = False
    

    #there must be a cleaner way to do this... can you assign directly to locals?
    backprop_args = args["backprop_args"]
    lr = backprop_args["lr"]# float = 0.05, 
    plott = backprop_args["plott"] #: bool = False, 
    reg = backprop_args.get("reg")
    plot_every_n_epochs = backprop_args["plot_every_n_epochs"]#: int = 2000, 
    SAVE_AFTER_EPOCHS = backprop_args["SAVE_AFTER_EPOCHS"]
    force = backprop_args.get("force")
    spikethreshold = backprop_args.get("spikethreshold")
    force_t = backprop_args.get("force_t")
    custom_loss = args["custom_loss"]
    epochs = args["epochs"]
    new_X = args["New_X"]
    states_dot = args["states_dot"]
    LinOut = args["out_W"]
    force_t = args["force_t"]
    criterion = args["criterion"]
    #spikethreshold = args["spikethreshold"]
    t = args["t"]
    g, g_dot = args["G"]
    gamma_cyclic = args["gamma_cyclic"]
    gamma = args["gamma"]
    init_conds = args["init_conds"]
    ode_coefs = args["ode_coefs"]
    enet_strength = args["enet_strength"]
    enet_alpha = args["enet_alpha"]
    init_conds[0] = y0

    optimizer = optim.Adam([LinOut.weight, LinOut.bias], lr = lr)
    if gamma:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    else:
        scheduler = None


    if gamma_cyclic:
        cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 10**-6, 0.01,
                                            gamma = gamma_cyclic,#0.9999,
                                            mode = "exp_range", cycle_momentum = False)
    if plott:
      #use pl for live plotting
      fig, ax = pl.subplots(1,3, figsize = (16,4))




    loss_history = []
    lrs = []

    #float(loss) of the previous iteration: previous_loss
    floss_last = 0
    best_score = 10000
    pow_ = -4

    best_bias, best_weight, best_fit, best_ydot = [None * 4]


    with torch.enable_grad():
        
        #begin optimization loop
        for e in range(epochs):

            optimizer.zero_grad()

            N = LinOut( new_X)
            N_dot = states_dot @ LinOut.weight.T
            y = g * N 

            ydot = g_dot * N + g * N_dot

            for i in range(y.shape[1]):
                y[:,i] = y[:,i] + init_conds[i]

            #old assert statements for debugging:
            #assert N.shape == N_dot.shape, f'{N.shape} != {N_dot.shape}'
            #assert esn.LinOut.weight.requires_grad and esn.LinOut.bias.requires_grad

            #old regularization code:
            #total_ws = esn.LinOut.weight.shape[0] + 1
            #weight_size_sq = torch.mean(torch.square(esn.LinOut.weight))

            loss = custom_loss(t, y, ydot, LinOut.weight, reg = reg, ode_coefs = ode_coefs,
                    init_conds = init_conds, enet_alpha= enet_alpha, enet_strength = enet_strength, force_t = force_t)
            
            #log the loss
            floss = float(loss)
            loss_history.append(floss)

            #backwards pass
            loss.backward()
            
            #optimizer adjustments:
            optimizer.step()
            if gamma_cyclic and e > 100 and e <5000:
                cyclic_scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])
            
            #The parameter spikethreshold is very effective when getting the RC to converge durring unsupervised solving of non-linear diffeqs.
            if e > 0:
                log_loss_delta = float(np.log(floss_last) - np.log(floss)) 
                if log_loss_delta > spikethreshold:
                    lrs.append(optimizer.param_groups[0]["lr"])
                    if scheduler != None:
                        scheduler.step()


            #save the best scoring model along with the fit, the bias, and the weights. 
            if e > SAVE_AFTER_EPOCHS:

            	#if we haven't yet saved a best score (prevent crashes)
                if not best_score:
                    best_score = min(loss_history)
                
                #if the loss is the lowest yet observed overwrite the best model properties
                if floss < best_score:  
                    best_bias, best_weight = LinOut.bias.detach(), LinOut.weight.detach()
                    best_score = float(loss)
                    best_fit = y.clone()
                    best_ydot = ydot.clone()
            # else:
            #     best_bias, best_weight, best_fit = LinOut.bias.detach(), LinOut.weight.detach(), y.clone()

            floss_last = floss

            #Old code to terminate early:
            
            # if e >= EPOCHS_TO_TERMINATION and EPOCHS_TO_TERMINATION:
            #     return {"weights": best_weight, "bias" : best_bias, "y" : best_fit, 
            #           "loss" : {"loss_history" : loss_history},  "best_score" : torch.tensor(best_score),
            #           "RC" : esn}
            
            #deactivated plotting code:

            # if plott and e:
            #     if e % plot_every_n_epochs == 0:
            #         for param_group in optimizer.param_groups:
            #             print('lr', param_group['lr'])
            #         ax[0].clear()
            #         logloss_str = 'Log(L) ' + '%.2E' % Decimal((loss).item())
            #         delta_loss  = ' delta Log(L) ' + '%.2E' % Decimal((loss-floss_last).item())

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
            #         floss_last = float(loss.item())

            #         #clear the plot outputt and then re-plot
            #         display.clear_output(wait=True) 
            #         display.display(pl.gcf())


    return {"weights": best_weight, "bias" : best_bias, "y" : best_fit, "ydot" : best_ydot, 
          "loss" : {"loss_history" : loss_history}, "best_score" : torch.tensor(best_score)}
          #"RC" : esn}