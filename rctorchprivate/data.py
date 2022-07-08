import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import torch

#from pyESN import ESN
from matplotlib import pyplot as plt
from numpy import loadtxt
from scipy.integrate import odeint
import pickle
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

class Fpendulum:
    #was trained on A=0.5, W =0.2
    """forced pendulum"""
    def __init__(self, t, x0,  px0, lam=1, A=0, W=1, force = "sin"):
        """
        Arguments
        ---------
        t  : dtype
            desc
        x0 : float
            initial position
        px0 : float
            initial momentum
        lam : float
            ...
        A : float
            ...
        W : float
            ...
        force : str
            "sin", "cos", or "sincos"
        """

        self.t = t
        self.u0 = [x0, px0]
        # Call the ODE solver
        if force == "sin":
            self.force =  Fpendulum.sin_force
            spec_f = Fpendulum.sin_f
        elif force == "cos":
            self.force = Fpendulum.cos_force
            spec_f = Fpendulum.cos_f
        elif force == "sincos":
            self.force =  Fpendulum.sincos_force
            spec_f = Fpendulum.sincos_f
        solPend = odeint(spec_f, self.u0, t, args=(lam,A,W))
        self.xP = solPend[:,0];        
        self.pxP = solPend[:,1]; 
        self.force_pend_data = solPend
        
        self.lineW = 3
        self.lineBoxW=2
        
        #self.force = self.force(A, W, t)

        #         self.font = {'family' : 'normal',
        #                 'weight' : 'normal',#'bold',
        #                 'size'   : 24}

        #plt.rc('font', **font)
        #plt.rcParams['text.usetex'] = True
    
    @staticmethod
    def sin_force(A, W, t):
        return A*np.sin(W*t)
    
    @staticmethod
    def sincos_force(A, W, t):
        if isinstance(W, int) or isinstance(W, float):
            return A*np.sin(W*t)*np.cos(W*t)
        elif isinstance(W, list):
            W1, W2 = W
            return A*np.sin(W1*t)*np.cos(W2*t)
    
    @staticmethod
    def cos_force(A, W, t):
        return A*np.cos(W*t)
    
    @staticmethod
    def cos_f(u, t, lam=0,A=0,W=1,gamma=0, w=1):
        
        x,  px = u        # unpack current values of u
        derivs = [px, -gamma * px - np.sin(x) + Fpendulum.cos_force(A, W, t)]     #     you write the derivative here
        return derivs
    
    @staticmethod
    def sin_f(u, t, lam=0,A=0,W=1,gamma=0, w=1):
        
        x,  px = u        # unpack current values of u
        derivs = [px, -gamma * px - np.sin(x) + Fpendulum.sin_force(A, W, t)]     #     you write the derivative here
        return derivs
    
    @staticmethod
    def sincos_f(u, t, lam=0,A=0,W=1,gamma=0, w=1):
        
        x,  px = u        # unpack current values of u
        derivs = [px, -gamma * px - np.sin(x) + Fpendulum.sincos_force(A, W, t)]     #     you write the derivative here
        return derivs
    
    def return_data(self):
        return np.vstack((my_fp.xP, my_fp.pxP)).T
    
    def plot_(self):
        """
        """
        plt.figure(figsize=[20,6])
        plt.subplot(1,2,1)
        plt.plot(self.t/np.pi, self.xP,'b',label='x(t)', linewidth = self.lineW)
        plt.plot(self.t/np.pi, self.pxP,'r',label='v(t)', linewidth = self.lineW)
        plt.xlabel('$t/\pi$')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.xP, self.pxP,'g', linewidth = self.lineW)
        plt.xlabel('$x$')
        plt.ylabel('$v$')

pi = 3.14159265358979323846264338327950288419716939937510

dl = 60.0* pi

dt__ = pi/200

def synthetic_data(desired_length =dl, 
                   force = "sin", 
                   dt = dt__, 
                   AW_lst = None,
                   x0_p0_lst = [ (0.1, 0.1), (0.25, 0.25), (0.3, 0.45),  (0.5, 0.5)]):
    """Generates synthetic data

    Arguments
    ---------
    desired_length : float
        the desired maximum value of the synthetic data
    force : str
        can be "sin", "cos", or "sincos"
    dt : float
        the size of the discrete step between sequence element i and i + 1
    AW_list : list of tuples  (float elements)
        list of alpha and omega, [(alpha_1, omega_1), (alpha_2, omega_2), ...]
    x0_p0_lst : list of tuples (float elements)
        list of initial positions (x0) and inital momenta (p0)
        ie: [(x0_1, p0_1), (x0_2, p0_2), ...]


    this function should
    Returns
    -------
    datas, inputs, t : dtype1, dtype2, dtype3...

    datas is a list of lists, the individual elements are tensors
        this is a (-1, 2) shaped tensor where -1 is the length of the data
        the vector in the first column is the position, the vector in the second column is the momentum
    inputs is a list of lists, the individual elements are tensors
        this is a (-1, 2) shaped tensor where -1 is the length of the tensor
        ...
    t is a numpy array

    """
    
    #desired_length = 60*np.pi
    #dt = np.pi/200
    
     #

    datas = []
    inputs = []

    for i, (A, W) in enumerate(AW_lst):
        print(A, W)

        #for the experiments without noise use 60*np.pi

        Nt = int(desired_length//dt)
        t = np.linspace(0, desired_length, Nt) #100*np.pi

        trajectories_i = []
        traj_i_inputs = []

        for j, (x0, px0) in enumerate(x0_p0_lst):
        #x0, px0 =  #.3, .5

            my_fp = Fpendulum(t = t, x0 = x0, px0 = px0, A = A, W = W, force = force)
            force_pend_data = my_fp.force_pend_data
            input_ = my_fp.force(t = my_fp.t, A = A, W = W)

            force_pend_data = torch.tensor(force_pend_data, dtype = torch.float32)
            input_ = torch.tensor(input_, dtype = torch.float32)
            traj_i_inputs.append(input_)

            trajectories_i.append(force_pend_data)
        datas.append(trajectories_i)
        inputs.append(traj_i_inputs)
    return datas, inputs, t

def if_numpy(arr_or_tensor):
    if type(arr_or_tensor) == np.ndarray:
        arr_or_tensor = torch.tensor(arr_or_tensor) 
    return arr_or_tensor

class Splitter:
    
    def __init__(self, tensor, split = 0.6, noise = False, std = 0.07):
        self._split = split
        
        self._tensor = if_numpy(tensor).clone()
        self._std = std
        self._noise = noise
        if noise:
            self.make_noisy_data()
        #return self.split()
    
    def split(self):
        tl = len(self._tensor)
        trainlen = int(tl * self._split)
        train, test = self._tensor[:trainlen], self._tensor[trainlen:]
        return train, test

    def make_noisy_data(self):
        self._tensor += torch.normal(0, self._std, size = self._tensor.shape)
    
    def __repr__(self):
        strr = f"Splitter: split = {self._split},"
        if self._noise:
            strr += " noise = {self._std}"
        else:
            strr += " noise = False"
        return strr

def split_data(input_tensor, output_tensor, split):
    input_splitter = Splitter(input_tensor, split = split)
    input_tr, input_te = input_splitter.split()
    output_splitter = Splitter(output_tensor, split = split)
    target_tr, target_te = output_splitter.split()
    
    return input_tr, input_te, target_tr, target_te

def plot_data(data, force):
    fig, ax = plt.subplots(2,1, figsize=(16,7))
    #TODO confirm that position and momentum are correct
    ax[1].plot(data[:,0], label = "position")
    ax[1].plot(data[:,1], label = "momentum")
    ax[0].plot(force, '--', label = "force")
    ax[0].set_ylim(-1.3, 1.3) 
    ax[0].set_title("Force Observer")
    
    ax[1].set_title("Forced Pendulum target data")
    [ax[i].legend() for i in range(2)]
    plt.tight_layout()
    plt.show()

def final_figure_plot(test_gt, noisy_test_gt, rc_pred, 
                      color_noise = None, 
                      color_gt = None, 
                      color_rc = 'brown', 
                      gt_linestyle = None, 
                      pred_linestyle = None,
                      noisy_linestyle = ':', 
                      linewidth = 1,
                      label_fontsize = 18, 
                      title_fontsize = 18, 
                      plot_title = False, 
                      alpha = None,
                      legend = False, 
                      color_map = None,
                      noise_xlim = None, 
                      noise_ylim = None,
                      magma = False, 
                      noisy_alpha = None, 
                      noisy_s = None, 
                      figsize = (9,4)):
    """
    #Todo write docstrings

    example colormap: cm.gnuplot2_r(resids)
    """
    
    if magma:
        from matplotlib import cm
        from sklearn.preprocessing import MinMaxScaler

        resids = ((test_gt - rc_pred)**2)

        log_resids = np.log10(resids)
        

        assert log_resids.shape == test_gt.shape, f'{log_resids.shape} != {test_gt.shape}'

        

        scaler = MinMaxScaler()

        mean_log_resids = log_resids.mean(axis =1).reshape(-1,1)

        norm_mean_log_residuals = scaler.fit_transform(mean_log_resids)
        #log_resid_color = mean_log_resids - mean_log_resids.min() + 0.00001

        plt.hist(norm_mean_log_residuals)
        plt.show()
        color_map_ = color_map(norm_mean_log_residuals.ravel())
    def phase_plot(tensor_, label, alpha = 0.9, color = None, magma = magma, other_val = None, linestyle = None, s = None):
        arg_dict = {"label" : label,
                    "alpha" : alpha,
                    "linewidth" : linewidth,
                    "linestyle" : linestyle}



        x1, x2 = tensor_[:,0], tensor_[:,1]
        

        if not magma:
            #for the noisy data:
            if linestyle == "." :
                plt.scatter(x1, x2, alpha = arg_dict['alpha'], color = color, s = s)
            else:

                plt.plot(x1, x2, **arg_dict, color = color)
        elif magma:
            #resids = ((tensor_ - other_tensor)**2).mean(axis = 1)
            plt.scatter(x1, x2, c=color_map_, edgecolor='none')
        else:
            assert False, f'magma argument: {magma} must be a boolean.'
    
    #plot 1 is the real phase space of the  virgin target test set
    if noisy_test_gt is not None:
        
        tick_font_reduction = 3
        tick_fontsize = label_fontsize-tick_font_reduction
        
        fig, ax = plt.subplots(1, 3, figsize = figsize)
        
        #first plot
        plt.sca(ax[0])
        if plot_title:
            ax[0].set_title("Ground Truth", fontsize =title_fontsize)
        phase_plot(test_gt, "latent_gt", alpha = alpha, color = color_gt, magma = False, linestyle = gt_linestyle)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.ylabel(r"$p$",  fontsize=label_fontsize)
        

        #plot 2 is the noisy phase space of the target test set
        plt.sca(ax[1])
        if plot_title:
            ax[1].set_title("Data", fontsize = title_fontsize)
        
        phase_plot(noisy_test_gt, "noisy_gt", alpha = noisy_alpha, 
                    color = color_noise, magma = False, linestyle = '.', s = noisy_s)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        #plt.ylabel("momentum")
        
        #ax = plt.gca()
        #ax.axes.xaxis.set_visible(False)
        plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        #top=False,         # ticks along the top edge are off
                        labelsize = label_fontsize,
                        labelleft=False) 

        #plot 3 is the phase space of the rc
        plt.sca(ax[2])
        if plot_title:
            ax[2].set_title("RC prediction", fontsize = title_fontsize, linestyle = pred_linestyle)
        phase_plot(rc_pred, "RC", color = color_rc, linestyle = pred_linestyle)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        #plt.ylabel("momentum")
        
        plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        #top=False,         # ticks along the top edge are off
                        labelsize = label_fontsize,
                        labelleft=False) 
        
        #ax.axes.xaxis.set_visible(False)
        if noise_xlim:
            for i in range(3):
                ax[i].set_xlim(noise_xlim)
        
        if noise_ylim:
            for i in range(3):
                ax[i].set_ylim(noise_ylim)
        [ (plt.sca(ax[i]), plt.xticks(fontsize=tick_fontsize), plt.yticks(fontsize=tick_fontsize)) for i in range(3)]
        
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, 2, figsize = figsize)
        
        tick_font_reduction = 5
        tick_fontsize = label_fontsize-tick_font_reduction
        
        plt.sca(ax[0])
        
        phase_plot(test_gt, label = "Ground Truth", alpha = alpha, color = color_gt, magma = False, linestyle = gt_linestyle)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.ylabel(r"$p$",  fontsize=label_fontsize)
        
        plt.sca(ax[1])
        plt.tick_params(labelleft=False)
        phase_plot(rc_pred, label = "RC", alpha = alpha, color = color_rc, linestyle = pred_linestyle)
        #fig.colorbar(cm.ScalarMappable(cmap=color_map)) # ax=ax
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        
        tff = tick_fontsize
        #change the tick
        [ (plt.sca(ax[i]), plt.xticks(fontsize=tff), plt.yticks(fontsize=tff), plt.xticks(rotation=45)) for i in range(2)]
        
        
        plt.tight_layout()
        
        if plot_title:
            ax[0].set_title("Ground Truth", fontsize = title_fontsize)
            ax[1].set_title("RC Prediction", fontsize = title_fontsize)
    
    if legend:
        handles, labels = ax[0].get_legend_handles_labels()
        handles2, labels2 = ax[1].get_legend_handles_labels()
        print('handles type:', type(handles))

        lines = handles + handles2
        labels = labels + labels2

        fig.legend( lines, labels, loc = 'lower center')#, loc = (0.5, 0), ncol=5 )

def off_switch(axis):
    assert axis in ["y", "x", "both"], "Value error"

    if axis in ["x", "both"]:
        plt.tick_params(
                        axis=axis,          # changes apply to the x-axis or y-axis or both
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        #top=False,         # ticks along the top edge are off
                        labelbottom=False) 
    if axis in ["y", "both"]:
        plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left=False,      # ticks along the bottom edge are off
                    #top=False,         # ticks along the top edge are off
                    labelleft=False)          

def final_figure_plot2(test_gt, noisy_test_gt, rc_pred1, rc_pred2, 
                       color_noise = None, 
                       color_gt = None, 
                       color_rc = 'brown', 
                       alpha = None,
                       noisy_format = '.', 
                       linestyle = '--', 
                       pred_linestyle = "-.",
                       noise_linestyle = ".", 
                       linewidth = 1,
                       label_fontsize = 21, 
                       title_fontsize = 18, 
                       noisy_alpha = 0.4,
                       noisy_s = 1,
                       figsize = (9,4), 
                       legend = False, 
                       alphas = None,
                       noise_xlim = None,
                       noise_ylim = None,
                       color_map = None,
                       #"label_fontsize" : 20,
                       rotate_xaxis_label = True,
                       magma = False):
    
    # arg_dict = {#"label" : label,
    #             "alpha" : alpha,
    #             "linestyle" : linestyle,
    #             "linewidth" : linewidth}
    tick_font_reduction = 4
    tff = tick_fontsize = label_fontsize-tick_font_reduction
    
    def phase_plot(tensor_, label, alpha = 0.9, color = None, magma = magma, other_val = None, linestyle = None, s = None):
        arg_dict = {"label" : label,
                    "alpha" : alpha,
                    "linewidth" : linewidth,
                    "linestyle" : linestyle}



        x1, x2 = tensor_[:,0], tensor_[:,1]
        

        if not magma:
            #for the noisy data:
            if linestyle == "." :
                plt.scatter(x1, x2, alpha = arg_dict['alpha'], color = color, s = s)
            else:

                plt.plot(x1, x2, **arg_dict, color = color)
        elif magma:
            #resids = ((tensor_ - other_tensor)**2).mean(axis = 1)
            plt.scatter(x1, x2, c=color_map_, edgecolor='none')
        else:
            assert False, f'magma argument: {magma} must be a boolean.'

               


    
    #plot 1 is the real phase space of the  virgin target test set
    if noisy_test_gt is not None:
        
       
        fig, ax = plt.subplots(2, 2, figsize = figsize)
        ax = ax.flatten()
        plt.sca(ax[0])
        phase_plot(test_gt, "latent_gt", color = color_gt)
        #plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.ylabel(r"$p$",  fontsize=label_fontsize)

        off_switch('x')
        

        #plot 2 is the noisy phase space of the target test set
        plt.sca(ax[1])
        phase_plot(noisy_test_gt, "noisy_gt", alpha = noisy_alpha, color = color_noise, magma = False, linestyle = '.', s = noisy_s)
        
        #ax[1].set_title("Data", fontsize = title_fontsize)
        #plt.xlabel(r"$x$",  fontsize=label_fontsize)
        
        #turn off axis ticks and labels
        off_switch('both')

        #plot 3 is the phase space of the rc
        plt.sca(ax[2])
        phase_plot(rc_pred1, "rc_prediction", linestyle = pred_linestyle, color = color_rc)
        #ax[2].set_title("RC pure prediction", fontsize = title_fontsize)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.ylabel(r"$p$",  fontsize=label_fontsize)
        
#         plt.tick_params(
#                         axis='y',          # changes apply to the x-axis
#                         which='both',      # both major and minor ticks are affected
#                         left=False,      # ticks along the bottom edge are off
#                         #top=False,         # ticks along the top edge are off
#                         labelleft=False) 
        
        plt.sca(ax[3])
        phase_plot(rc_pred2, "rc_prediction", linestyle = pred_linestyle, color = color_rc)
        #ax[3].set_title("RC w/ driven force", fontsize = title_fontsize)
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        #plt.ylabel("momentum")
        
        off_switch('y')

        plt.tight_layout()

        if noise_xlim:
            for i in range(4):
                ax[i].set_xlim(noise_xlim)
        
        if noise_ylim:
            for i in range(4):
                ax[i].set_ylim(noise_ylim)
        [ (plt.sca(ax[i]), plt.xticks(fontsize=tff), plt.yticks(fontsize=tff)) for i in range(4)]
        
        if rotate_xaxis_label:
            [ (plt.sca(ax[i]), plt.xticks(rotation=45)) for i in range(4)]
        
        #plt.xticks(rotation=45)
    else:
        fig, ax = plt.subplots(1, 3, figsize = figsize)
        
        plt.sca(ax[0])
        phase_plot(test_gt, "latent_gt", color = color_gt, alpha = alphas[0])
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.ylabel(r"$p$",  fontsize=label_fontsize)
        
        plt.sca(ax[1])
        plt.tick_params(labelleft=False)
        phase_plot(rc_pred1, "rc_prediction",  linestyle = pred_linestyle, color = color_rc, alpha = alphas[1])
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.tight_layout()
        
        plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        labelleft=False) 
        
        plt.sca(ax[2])
        plt.tick_params(labelleft=False)
        phase_plot(rc_pred2, "rc_prediction", linestyle = noise_linestyle, color = color_rc, alpha = alphas[2])
        plt.xlabel(r"$x$",  fontsize=label_fontsize)
        plt.tight_layout()
        
        plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        labelleft=False) 
        
        # ax[0].set_title("Ground Truth", fontsize = title_fontsize)
        # ax[1].set_title("RC Prediction old hps", fontsize = title_fontsize)
        # ax[2].set_title("RC Prediction new hps", fontsize = title_fontsize)
        
    if legend:
        handles, labels = ax[0].get_legend_handles_labels()
        handles2, labels2 = ax[1].get_legend_handles_labels()
        print('handles type:', type(handles))

        lines = handles + handles2
        labels = labels + labels2

        fig.legend( lines, labels, loc = (0.5, 0), ncol=5 )
        
        #ax.axes.xaxis.set_visible(False)

@ray.remote(max_calls=1)
def evaluate_rcs(datasets, i, hps, observers = True):#, target_tr, target_te, obs_tr, obs_te):
    
    dataset = datasets[i]
    print(f'datasets: {dataset["a"]}')
    
    t, force, data = dataset["t"], dataset["force"], dataset["data"]

    #for now just do the pure pred.
    t_splitter = Splitter(t, noise = False)
    ttrain, ttest = t_splitter.split()

    data_splitter = Splitter(data, noise = False)
    target_train, target_test = data_splitter.split()
    
    force_splitter = Splitter(force, noise = False)
    input_train, input_test = force_splitter.split()
    
    #datasets[0]["force"]
    
    my_esn = RcNetwork(**hps, 
                    random_state = 210, 
                    feedback = 1,
                    activation_function = {"tanh" : 0.1, 
                           "relu" : 0.9, 
                           "sin": 0.05},)
    
    if observers:
        fit = my_esn.fit(X = input_train, 
                    y = target_train,
                    burn_in = 0) #gt_override=target_train)

        val_scores, pred_ = my_esn.test(X = input_test, 
                                        #gt_override=target_test,
                                        y = target_test)
    else:
        fit = my_esn.fit(X =  None, #input_train, 
                    y = target_train,
                    burn_in = 0) #gt_override=target_train)

        val_scores, pred_ = my_esn.test(X = None, #input_test,
                    #gt_override=target_test,
                    y = target_test)

    resids = ( pred_ - target_test) ** 2 #my_esn.te_resids

    #we need x and px eventually, this is fine for now
    max_resid  = torch.max(resids)
    mean_resid = torch.mean(resids)
    
    #tensors2save = {}
    
    data2save = {"observers" : False,
                 "max_resid" : max_resid,
                 "mean_resid" : mean_resid,
                 "a" : dataset["a"],
                 "w" : dataset["w"],
                 "resids" : resids,
                 "tr_target" : target_train.numpy(),
                 "tr_pred" : fit,
                 "te_target" :  target_test.numpy(),
                 "te_pred" : pred_.numpy()}
    return data2save#, tensors2save

def preprocess_parallel_batch(lst, batch_size):
    iters = []
    floor_div = len(lst)//batch_size
    remainder = len(lst) % batch_size
    for i in range(floor_div):
        iters.append(batch_size)
    if remainder != 0:
        iters += [remainder]
    start_index = 0
    batched_lst = []
    for iterr in iters:
        stop_index = start_index + iterr
        batched_lst.append(lst[start_index:stop_index])
    return batched_lst

def retrieve_dataset(dataset_obj, hps, batch_size = 9):
    
    #     range_ = range(0, max(len(dataset_obj.datasets) + 1, batch_size + 1), batch_size)
    rez = []
    datasets = dataset_obj.datasets.copy()
    datasets_id = ray.put(datasets)
    batch_indices = preprocess_parallel_batch(list(range(len(datasets))), batch_size)
    total_idx = 0
    
    hps_id = ray.put(hps)
    for i, sub_batch in enumerate(batch_indices):
        start = sub_batch[0] + total_idx
        stop = sub_batch[-1]+ total_idx
        
        print(f'start {start} stop {stop}')
        
        this_set_indexs = list(range(start, stop))
        #time.sleep(0.5)
        
        results = ray.get([evaluate_rcs.remote(datasets_id, i, hps_id) for i in this_set_indexs])
        
        rez+=results
        #         if i == 2:
        #             breakpoint()
        #evaluate_rcs_plain(datasets_spec[0], hps)
        total_idx += (stop - start)
        print(f'percent_complete {i/len(batch_indices) * 100}')
    return rez

def make_results_dfs(rez_, fp = None):
    
    
    full_data_keys = ["tr_target", "tr_pred", "te_target", "te_pred", "resids"]
    pd_keys = ["observers", "max_resid", "mean_resid", "a", "w"]
    
    #pd_results = pd.DataFrame(rez_[pd_keys])
    data_summaries = []
    
    for dict_ in rez:
        data_summaries.append({key: float(val) for key, val in dict_.items() if key in pd_keys})
    
    #     for results in rez:
    #         try :
    #             pd_results[col] = pd_results[col].astype(float)
    #         except:
    #             pass #assert False, f'{col} {pd_results[col]}'
    #     pd_results.head()
    if fp:
        pd_results.to_csv(fp)
    return pd.DataFrame(data_summaries), rez_

def load_and_view(force = "cos", 
                  val = "mean", 
                  observers = True, 
                  vmin = 10**(-4.5), 
                  vmax = 1, 
                  cbar = False, 
                  figsize = (5,5), 
                  ax = None, 
                  label_off = False, 
                  cbar_ax = None,
                  view = True, 
                  label_fontsize = 20, 
                  xlabel_off = False,
                  tick_fontsize = 20):
    """
    force: can be "sin", "cos", or "sincos"
    val: can be "mean" or "max"
    """
    tff =  tick_fontsize 
    if val == "mean":
        val_ = "mean_resid"
    if val == "max":
        val_ = "max_resid"
    
    if observers:
        
        fp = "new_results/" + force + "_results.csv"
        fp2 = "new_results/"+ force + "_all_data.pickle"
    else:
        fp = "new_results/no_obs/" + force + "_results.csv"
        fp2 = "new_results/no_obs/"+ force + "_all_data.pickle"

    pd_results_ = pd.read_csv(fp)
    pd_results_.head()
    
    with open(fp2, 'rb') as handle:
        cos_results = pickle.load(handle)
    if ax:
        plt.sca(ax)
        
    #plt.title(force + " force: " + val)
    pivot = pd_results_.pivot(index='a', columns='w', values=val_)
    
    if view:
        if ax is None:
            plt.figure(figsize= figsize)
        plt.sca(ax)
        g = sns.heatmap(pivot, vmin = vmin, vmax = vmax,  cbar_ax = cbar_ax,
                    norm=LogNorm(vmin = vmin , vmax = vmax), cmap = "rocket", cbar = cbar)
                    #zmin = zmin)
        

        #cmap = "cubehelix")  #)#np.log10(pivot))
        g.set_facecolor('royalblue')#'lightblue')#'lightpink')
        yticklabels = g.get_yticklabels()
        xticklabels = g.get_xticklabels()
        g.set_yticklabels([round(float(yticklabels[i].get_text()),3) for i in  range(len(yticklabels))])
        g.set_xticklabels([round(float(xticklabels[i].get_text()),3) for i in  range(len(xticklabels))])

        if xlabel_off:
            off_switch('x')
            plt.xlabel(None)
        else:
            plt.xlabel(r'$\omega$', fontsize=label_fontsize)

        if label_off:
            plt.tick_params(    labelsize = label_fontsize,
                                axis='y',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                left=False,      # ticks along the bottom edge are off
                                labelleft=False)
            plt.ylabel(None, fontsize = 1)
        else:
            plt.ylabel(r'$\alpha$', fontsize=label_fontsize)
            [ (plt.sca(ax[i]), plt.xticks(fontsize=tff), plt.yticks(fontsize=tff), plt.xticks(rotation=45)) for i in range(2)]
        

    #ylabels = ['{:,.2f}'.format(x) for x in g.get_yticks()]
    #g.set_yticklabels(ylabels)
    #plt.show()
    return cos_results


class Noise_comp:
    """ A function that does noise comparison. Many noise levels for different stuff.
    
    """
    def __init__(self, noise_start, noise_stop, noise_step):
        self.noise_levels = list(np.arange(noise_start, noise_stop, noise_step))
        self.columns = ["x", "xP", "noise", "t"]
        self.count = 0
        
    def run_experiment(self, observers):
        for noise in self.noise_levels:
            print("noise", noise)
            for i in range(1):
                #get the data
                exp2_splitter = Splitter(force_pend_data, noise = True, std = noise)
                a, b = exp2_splitter.split()

                #fit the esn
                if observers:
                    my_esn = My_esn(**noise_hps_obs, 
                            random_state = 210, 
                            feedback = 1,
                            n_inputs = 1,
                            n_outputs = 2)
                    my_esn.fit(X =  None,
                                y = a, 
                                burn_in = 0, gt_override=target_train)
                    my_esn.test(X = None,
                                gt_override = target_test,
                     
                                y = b)
                else:
                    my_esn = My_esn(**noise_hps, 
                            random_state = 210, 
                            feedback = 1,
                            n_inputs = 1,
                            n_outputs = 2)
                    my_esn.fit(X =  input_train[:,1].view(-1,1),
                                y = a, 
                                burn_in = 0, gt_override=target_train)
                    my_esn.test(X = input_test[:,1].view(-1,1),
                                gt_override = target_test,
                                y = b)
                this_round_tr = my_esn.tr_resids
                this_round_te = my_esn.te_resids

                noise_tr = torch.ones_like(this_round_tr[:, 0].reshape(-1,1)).numpy() * noise
                noise_te = torch.ones_like(this_round_te[:, 0].reshape(-1,1)).numpy() * noise

                noise_tr =  np.round(noise_tr, 4)
                noise_te = np.round(noise_te, 4)

                this_round_tr = this_round_tr.numpy()
                this_round_te = this_round_te.numpy()
                #assert False, f'this_round_tr {this_round_tr.shape}this_round_te {this_round_te.shape}'

                t_tr = np.array(range(len(noise_tr))).reshape(-1,1)/40
                t_te = np.array(range(len(noise_te))).reshape(-1,1)/40

                if self.count == 0:
                    print("building d")
                    Data_tr = pd.DataFrame(this_round_tr)
                    Data_te = pd.DataFrame(this_round_te)
                    Data_tr["noise"] = noise_tr
                    Data_te["noise"] = noise_te

                    Data_tr["t"] = t_tr
                    Data_te["t"] = t_te
                    
                    Data_tr.columns = self.columns
                    Data_te.columns = self.columns

                else:
                    new_data_tr = np.concatenate((this_round_tr, noise_tr, t_tr), axis = 1)
                    new_data_te = np.concatenate((this_round_te, noise_te, t_te), axis = 1)

                    new_data_tr = pd.DataFrame(new_data_tr)
                    new_data_te = pd.DataFrame(new_data_te)

                    new_data_tr.columns = self.columns
                    new_data_te.columns = self.columns

                    Data_tr = pd.concat((Data_tr, new_data_tr), axis = 0)
                    Data_te = pd.concat((Data_te, new_data_te), axis = 0)

                self.count +=1
        self.Data_tr = Data_tr
        self.Data_te = Data_te
        
    def loss_plot(self, var = "x", data = "te"):
        if data == "te":
            df = self.Data_tr
        elif data == "tr":
            df = self.Data_te
        
        fig, ax = plt.subplots(1,1, figsize = (16, 4))
        self.g = sns.lineplot( x = 't', y = var, data = df, ax = ax, hue = "noise")
        plt.yscale('log')
        xlabels = ['{:,.4f}'.format(x) for x in self.g.get_xticks()/10]
        self.g.set_xticklabels(xlabels)
        plt.show()

class Fp_DataGen:
    """ A class that generates and stores force_pend data"""
    
    def __init__(self, A_range, W_range, 
                 Nt = 20000, length = 100*pi, 
                 x0= .5, px0 = .5, dt = None, split = 0.6, non_resonant_only = True,
                 threshold = 5, force = "sin"):
        """
        Arguments:
            Nt: number of time points
            length: number of datapoints
            dt: upgrade later so we can take dt instead of Nt and length
            x0, px0: initial position and momentum
            A_range: the range of alpha (should be np.arrange)
            W_range: the range of W  (should be np.arrange)
        """
        #original was 8000, and 40 pi
        #new is  20000 and 100 pi to preserve dt
        
        self.x0, self.px0 = x0, px0
        t = np.linspace(0, length, Nt)
        self.datasets = []
        
        for i, a in enumerate(A_range):
            for j, w in enumerate(W_range):
                my_fp = Fpendulum(t = t, x0 = x0, px0 = px0, A = a, W = w, force = force)
                data = my_fp.force_pend_data
                force_ = my_fp.force(A = a, W = w, t = t)
                t = my_fp.t
                
                if np.max(np.abs(data)) > threshold:
                    #resonant.append(1)
                    resonant = 1
                else:
                    resonant = 0
                
                fp_data_spec = {"a": a, "w": w, 
                                "data" : data,
                                "force" : force_,
                                "t" : t,
                                "resonant" : resonant}
                
                #enforce typing:
                for key, val in fp_data_spec.items():
                    if key != "resonant":
                        fp_data_spec[key] = torch.tensor(val, dtype = torch.float32)
                if non_resonant_only:
                    if resonant == 0:
                        self.datasets.append(fp_data_spec)
                else:
                    self.datasets.append(fp_data_spec)
        self.find_resonance()
        
    def plot_all(self):
        for dictt in self.datasets:
            plt.plot(dictt["data"], alpha = 0.1)
    
    def find_resonance(self, threshold = 10):
        resonant = []
        for i, dictt in enumerate(self.datasets):
            if torch.max(torch.abs(dictt["data"])) > threshold:
                resonant.append(1)
                self.datasets[i]["resonant"] = 1
            else:
                self.datasets[i]["resonant"] = 0
                resonant.append(0)
        
        return torch.tensor(resonant, dtype = torch.int32).reshape(-1,1)
    
    @staticmethod
    def _find_resonance(self, datasets, threshold = 10):
        resonant = []
        for i, dictt in enuemerate(datasets):
            
            if torch.max(torch.abs(dictt["data"])) > threshold:
                
                resonant.append(1)
            else:
                
                resonant.append(0)
        
        return torch.tensor(resonant, dtype = torch.int32).reshape(-1,1)
    
    def plot_resonant(self, threshold):
        plt.figure(figsize = (16, 4))
        rez = self.find_resonance(threshold)
        flag = 0
        for i, resonant_bool in enumerate(rez):
            if resonant_bool == 1:
                abs_data = torch.abs(self.datasets[i]['data'])
                abs_px = abs_data[:,0]
                abs_x = abs_data[:,1]
                if flag == 0:
                    plt.plot(abs_x, color = "red", alpha = 0.1, label = "x")
                    plt.plot(abs_px, color = "blue", alpha = 0.1, label = "px")
                    flag = 1
                else:
                    plt.plot(abs_x, color = "red", alpha = 0.1)
                    plt.plot(abs_px, color = "blue", alpha = 0.1)
                    
        plt.axhline(y = threshold, color = "black", label = "threshold")
        plt.yscale('log')
        plt.legend()
        plt.title("resonant")
        plt.show()
        
        for i, resonant_bool in enumerate(rez):
            if resonant_bool == 0:
                data = self.datasets[i]['data']
                abs_px = data[:,0]
                abs_x = data[:,1]
                plt.plot(abs_x, color = "red", alpha = 0.1)
                plt.plot(abs_px, color = "blue", alpha = 0.1)
        plt.legend()
        plt.title("non-resonant")
        plt.show()

def preprocess_parallel_batch(lst, batch_size):
    iters = []
    floor_div = len(lst)//batch_size
    remainder = len(lst) % batch_size
    for i in range(floor_div):
        iters.append(batch_size)
    if remainder != 0:
        iters += [remainder]
    start_index = 0
    batched_lst = []
    for iterr in iters:
        stop_index = start_index + iterr
        batched_lst.append(lst[start_index:stop_index])
    return batched_lst


#the following two functions were located right above `act_f_list` (individual experiment to test output_f vs multi-activations)
def individual_experiment(observer: bool = False,
                          f_out: bool = False,
                          random_state: int = 210,
                          multi = None
                         ): 
    input_tr = input_train if observer else None
    input_te = input_test if observer else None
    
    out_f = "tanh" if f_out else "identity"
    
    rc = RcNetwork(**opt_hps, 
                           output_activation = out_f,
                           activation_function = multi,
                           random_state = random_state, 
                           feedback = True)
    rc.fit(X = input_tr, 
                  y = target_train,
                  burn_in = 0) 
    score, prediction = rc.test(X = input_te, y = target_test)
    
#     rc.combined_plot()
#     final_figure_plot(target_test, None, prediction )
    print(f'score: {score}')
    return score


def activation_experiment(random_states, act_f_list, save_path = None):
    results = {"Score" : [], "Observer" : [], "f_out" : [],"random_state" : [], "multi" : []}
    for observer in [True, False]:
        for f_out in [True, False]:
            for random_state in random_states:
                for i, act_f_dict in enumerate(act_f_list):
                    score = individual_experiment(observer, f_out, random_state, act_f_dict)
                    results["Score"].append(float(score))
                    results["Observer"].append(observer)
                    results["f_out"].append(f_out)
                    results["random_state"].append(random_state)
                    results["multi"].append(i)
    results = pd.DataFrame(results)
    #save line: pd.save...
    return results


def run_experiments(datas, inputs, hps, split, output_activation,  noise = None, activation_dict = "tanh"):
    #split = 0.6
    scores = []
    for i, data in enumerate(datas):
        for j, trajectory in enumerate(data):
            force_pend_data__ = trajectory

            input_ = inputs[i][j]
            input_tr, input_te, target_tr, target_te = split_data( input_, trajectory, split)
           


            input_train, input_test, target_train, target_test = split_data( input_, force_pend_data, split)
            #input_train, input_test, target_train, target_test = split_data( input_, force_pend_data, 0.6)
            

            esn_pure_pred = RcNetwork(**hps, 
                        random_state = 210, 
                        feedback = True,
                        output_activation = output_activation,
                        activation_function = activation_dict)
                        #solve_sample_prop = 0.8
                                            
            if noise is not None:
                exp2_splitter = Splitter(trajectory, noise = True, std = noise, split = split)
                noisy_tr_target, noisy_te_target = exp2_splitter.split()
                
                esn_pure_pred.fit(X = noisy_tr_target, #input_tr,#None, 
                            y = target_tr, 
                            burn_in = 0)
                score, prediction = esn_pure_pred.test(X = noisy_te_target,
                            y = target_te)
                final_figure_plot(test_gt = target_te, noisy_test_gt = noisy_te_target, rc_pred = prediction, 
                      noisy_format = '.')
            else:
                esn_pure_pred.fit(X = input_tr, #input_tr,#None, 
                            y = target_tr, 
                            burn_in = 0)
                score, prediction = esn_pure_pred.test(X = input_te,
                            y = target_te)
                final_figure_plot(test_gt = target_te, noisy_test_gt = None, rc_pred = prediction, 
                      noisy_format = '.')
                
#             esn_pure_pred.combined_plot(gt_tr_override=target_tr,
#                                         gt_te_override=target_te)
            
            print(f'score : {score}')
            scores.append(score)
            
            plt.show()
    total_score = np.mean(scores)
    print(f'total score {total_score}')



def load(dataset_name : str, train_proportion : float = 0.5, dt = pi/200):
    """
    Arguments
    ---------
    dataset_name : str
        name of the dataset to include. 
    """
    dataset = {}


    valid_names = ['hennon_hailes', 'forced_pendulum']
    err_msg = f'Invalid entry, please make sure dataset is one of {valid_names}'

    assert dataset_name in valid_names, err_msg 

    if dataset_name == 'forced_pendulum':

        problem = 3
        idx = 0
        base = np.exp(1)

        As = np.ones(5)/2
        #omegas
        Ws = np.array([base**i for i in [2, 1, -0.95, -1, -2, -3]])

        
        #AW_lst = [(0, 0), (0.2, 0.2),  (0.3, 0.3), (0.1, 0.01)]
        AW_lst = list(zip(As, Ws))

        datas, inputs, t = synthetic_data(desired_length=np.pi * 60, 
                                          AW_lst = AW_lst,
                                          dt = dt)
        input_train, input_test, target_train, target_test = split_data(inputs[problem][idx], datas[problem][idx], train_proportion) #0.2
        plot_data(datas[problem][idx], force = input_train)

        dataset['force']  = input_train, input_test
        dataset['target'] = target_train, target_test
        return dataset
    pass
        
##########

#EXTRA Experiments from BO section of Tools of AI paper, probably don't need but kept here just in case.


# %%time
# #was 0.1 noise = 0.1

# experiment0_hps  = {'connectivity': 0.010497199729654356,
#  'spectral_radius': 1.6224205493927002,
#  'n_nodes': int(100.01666259765625),
#  'regularization': 0.019864175793163488,
#  'leaking_rate': 0.044748689979314804,
#  'bias': 0.8152865171432495}


# esn_pure_pred = RcNetwork(**experiment0_hps, activation_function = {'tanh':0.1, 'relu':0.9},
#                           feedback = True,
#                           random_state = 210) # was 210
# esn_pure_pred.fit(X = input_tr, y = target_tr)
# score, prediction = esn_pure_pred.test( X = input_te, y = target_te)
# esn_pure_pred.combined_plot()
# score

########

# #these hps are new
# experiment0_hps  = {'connectivity': 0.54275345999343795,
#  'spectral_radius': 1.5731128454208374,
#  'n_nodes': 251,
#  'regularization': 0.1880535350594487,
#  'leaking_rate': 0.03279460221529007,
#  'bias': 0.8625065088272095} 



# esn_pure_pred = RcNetwork(**experiment0_hps, 
#                           activation_function = {"relu" : 0.33, "tanh" : 0.5, "sin" : 0.1},
#                           feedback = True,
#                           random_state = 210) # was 210
# esn_pure_pred.fit(X = input_tr, y = target_tr)
# score, prediction = esn_pure_pred.test( X = input_te, y = target_te)
# esn_pure_pred.combined_plot()
# final_figure_plot(target_te, None, prediction)
# score


########## Walkthrough part 2 extra experiments:


# # set up the data
# problem, idx = 0, 1
# input_tr, input_te, target_tr, target_te = split_data( inputs[problem][idx], datas[problem][idx], 0.2) #
# noise = 0.4
# #was 0.1

# noise_target_tr = target_tr + torch.rand_like(target_tr)*noise
# noise_target_te = target_te + torch.rand_like(target_te)*noise

# ### with the original experiments:


# orig_hps = {'connectivity': 0.4071449746896983,
#  'spectral_radius': 1.1329107284545898,
#  'n_nodes': 202,
#  'regularization': 1.6862021450927922,
#  'leaking_rate': 0.009808523580431938,
#  'bias': 0.48509588837623596}

# unopt_rc = RcNetwork(**orig_hps, 
#                         #activation_function = {"relu" : 0.33, "tanh" : 0.5, "sin" : 0.1},
#                         feedback = True,
#                         random_state = 210) # was 210
# unopt_rc.fit(X = input_tr, y = noise_target_tr)
# score, prediction = unopt_rc.test( X = input_te, y = noise_target_te)
# score
# unopt_rc.combined_plot(gt_tr_override = target_tr, gt_te_override = target_te)
# print(f"score: {score}")
# final_figure_plot(target_te, noise_target_te, prediction)

### now with BO optimized hps
# noisep2_hps = {'connectivity': 0.08428374379805789,
#  'spectral_radius': 1.1395107507705688,
#  'n_nodes': int(251.67526245117188),
#  'regularization': 1.3896905679295795,
#  'leaking_rate': 0.06542816758155823,
#  'bias': 0.24717366695404053}

# rc_noise_p2 = RcNetwork(**noisep2_hps, 
#                         #activation_function = {"relu" : 0.33, "tanh" : 0.5, "sin" : 0.1},
#                         feedback = True,
#                         random_state = 210) # was 210
# rc_noise_p2.fit(X = input_tr, y = noise_target_tr)
# score, prediction = rc_noise_p2.test( X = input_te, y = noise_target_te)
# score

# rc_noise_p2.combined_plot(gt_tr_override = target_tr, gt_te_override = target_te)
# print(f"score: {score}")
# final_figure_plot(target_te, noise_target_te, prediction)



####################

### Alternative HPs: (500 instead of 200)

# %%time
# hps = {'n_nodes': 500,
#  'connectivity': 0.6475559084256522,
#  'spectral_radius': 1.0265705585479736,
#  'regularization': 61.27292863528506,
#  'leaking_rate': 0.010949543677270412,
#  'bias': 0.5907618999481201,
#       }

# # extra_hps = {"input_connectivity" : 0.2,
# #             "feedback_connectivity" : 0.2}

# esn_pure_pred = RcNetwork(**hps,
#             activation_function = ["tanh"],
#             random_state = 210, 
#             feedback = True)
# esn_pure_pred.fit(X = input_train, 
#             y = target_train, 
#             burn_in = 0)
# score, _ = esn_pure_pred.test(X = input_test,
#             y = target_test)
# esn_pure_pred.combined_plot()
# print(f'score: {score}')

# final_figure_plot(target_test,  None , prediction )
