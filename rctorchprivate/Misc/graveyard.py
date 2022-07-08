# I've put all old vestigal code here from rc.py. For cleanliness purposes:

"""
vestigal code from __init__ method

        TODO: additional hyper-parameters
        noise from pyesn — unlike my implimentation it happens outside the activation function. 
        TBD if this actually can improve the RC.
        self.PyESNnoise = 0.001
        self.external_noise = torch.rand(self.n_nodes, device = self.device)
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
        }"""
        #print("finished building RC")

"""
                                    #calculate F depending on the order of the ODE:
                                    if ODE == 1:
                                        #population eq
                                        #RHS = lam * self.y0 - f(self.X) 
                                        #self.F =  g_dot * states_  +  g * (states_dot + lam * states_)
                                        
                                        #nl eq 
                                        self.F =  g_dot * states_  +  g * states_dot
                                        if nl_f:
                                            #y0_nl, y0_nl_dot = nl_f(self.y0)
                                            self.F = self.F - 2 * self.y0 * g * states_ 
                                            #self.F = self.F - (g * ).T @ 

                                    elif ODE == 2:
                                        # without a reparameterization
                                        #self.F = torch.square(self.X) * states_dot2 + 4 * self.X * states_dot + 2 * states_ + (self.X ** 2) * states_
                                        self.G = g * states_
                                        assert self.G.shape == states_.shape, f'{self.shape} != {self.states_.shape}'
                                        self.Lambda = g.pow(2) * states_ 
                                        self.k = 2 * states_ + g * (4*states_dot - self.G*states_) + g.pow(2) * (4 * states_ - 4 * states_dot + states_dot2)
                                        self.F = self.k + self.Lambda
                                    #common F derivation:
                                    F = self.F.T
                                    F1 = F.T @ F 
                                    F1 = F1 + self.regularization * torch.eye(F1.shape[1], **self.dev)
                                    ##################################### non-linear adustment
                                    nl_adjust = False
                                    if nl_adjust:
                                        G = g * states_
                                        G_sq = G @ G.T
                                        nl_correction = -2 * self.y0 * (G_sq)
                                        F1 = F1 + nl_correction
                                    #F1_inv = torch.pinverse(F1)
                                    #F2 = torch.matmul(F1_inv, F.T)
                                    #####################################
                                    #First Order equation
                                    if self.ODE_order == 1:
                                        self.y0I = (self.y0 ** 2) * torch.ones_like(self.X)
                                        #self.y0I = self.y0I.squeeze().unsqueeze(0)
                                        #RHS = lam*self.y0I.T - f(self.X) 

                                        #REPARAM population
                                        #RHS = lam * self.y0 - f(self.X) 

                                        RHS = self.y0I

                                        #weight = torch.matmul(-F2.T, RHS)

                                        weight = torch.matmul(F2.T, RHS)
                                        #assert False, weight.shape

                                    #Second Order equation
                                    elif self.ODE_order == 2:
                                        
                                        #self.y0I = y0[0] * torch.ones_like(self.X)
                                        #self.y0I = self.y0I.squeeze().unsqueeze(0)

                                        #RHS = self.y0I.T + self.X * y0[1]
                                        RHS = self.y0 + f_t * y0[1]
                                        
                                        #t = self.X
                                        #A0 = y0 + g * v0
                                        #RHS = A0 + (g - 1)*v0 - f(t)
                                        weight = torch.matmul(-F2.T, D_A)
                                    weight = torch.matmul(D_W, D_A)

                                    #y = y0[0] + self.X * y0[1] + self.X
"""
"""
        if self.ODE_order == 1:
            return self.reparam(t = self.X, init_conditions = self.y0, N = self.yfit, N_dot = N_dot)
        elif self.ODE_order == 2:
            N_dot2 = self.calc_hdot(states_dot2[:,1:], cutoff = False)
            return self.reparam(t = self.X, init_conditions = [y0, v0], 
                N = self.yfit, N_dot = [N_dot, N_dot2], esn = self, 
                states = states_[:,1:], states_dot = states_dot[:,1:], states_dot2 = states_dot2[:,1:])
        """
"""
    #assert weight.requires_grad, "weight doesn't req grad"
                            #torch.solve solves AX = B. Here X is beta_hat, A is ridge_x, and B is ridge_y
                            #weight = torch.solve(ridge_y, ridge_x).solution
                        # elif self.l2_prop == 1:
                        # else: #+++++++++++++++++++++++         This section is elastic net         +++++++++++++++++++++++++++++++

                        #     gram_matrix = torch.matmul(train_x.T, train_x) 

                        #     regr = ElasticNet(random_state=0, 
                        #                           alpha = self.regularization, 
                        #                           l1_ratio = 1-self.l2_prop,
                        #                           selection = "random",
                        #                           max_iter = 3000,
                        #                           tol = 1e-3,
                        #                           #precompute = gram_matrix.numpy(),
                        #                           fit_intercept = True
                        #                           )
                        #     print("train_x", train_x.shape, "_____________ train_y", train_y.shape)
                        #     regr.fit(train_x.numpy(), train_y.numpy())

                        #     weight = torch.tensor(regr.coef_, device = self.device, **self.dev)
                        #     bias =  torch.tensor(regr.intercept_, device =self.device, **self.dev)


#if not preloaded_states_dict:
                # else:
                #     sd = preloaded_states_dict
                #     self.states, self.states_dot, G, self.extended_states = sd["s"], sd["s1"], sd["G"], sd["ex"]
                #     states_with_bias, states_dot_with_bias = sd["sb"], sd["sb1"]
                #     # if self.ODE_order == 2:
                #     #     self.states_dot2 = sd["s2"]
                #     #     states_dot2_with_bias = sd["sb2"]
                #     g, g_dot = G
                #     self.g = gdef
"""
#EchoStateNetwork = RcNetwork


# class Recurrence(Function):
#     """
#     Summary line.

#     Extended description of function.

#     Parameters
#     ----------
#     arg1 : int
#         Description of arg1
#     arg2 : str
#         Description of arg2

#     Returns
#     -------
#     int
#         Description of return value

#     """

#     @staticmethod
#     def forward(ctx, states, esn, X, y, weights):
#         """
#         Summary line.

#         Extended description of function.

#         Parameters
#         ----------
#         ctx : dtype
#             Description of arg1
#         states : dtype
#             Description of arg2
#         esn : RcNetwork
#             the echo-state network object
#         X : pytorch.tensor or numpy.array
#             observers
#         y : pytorch.tensor or numpy.array
#             response
#         weights : ... 
#             Desc
#         Returns
#         -------
#         states, states_dot : pytorch.tensor, pytorch.tensor
#             The hidden states and the derivative of the hidden states
#         """
#         states, states_dot = esn.train_states(X, y, states)
#         ctx.states = states
#         ctx.states_dot = states_dot
#         return states, states_dot
#     @staticmethod
#     def backward(ctx, grad_output, weights):
#         """
#         Summary line.
#         Extended description of function.
#         Parameters
#         ----------
#         ctx : int
#             Description of arg1
#         grad_output : str
#             Description of arg2
#         weights : pytorch.tensor?
#             Desc
#         Returns
#         -------
#         int
#             Description of return value
#         """
#         if grad_output is None:
#             return None, None
#         output = torch.matmul(ctx.states_dot, weights.T)
#         return output, None, None, None, None
                                    



    # def predict_stepwise(self, y, x=None, steps_ahead=1, y_start=None):
    #     """Predicts a specified number of steps into the future for every time point in y-values array.
    #     E.g. if `steps_ahead` is 1 this produces a 1-step ahead prediction at every point in time.
    #     Parameters
    #     ----------
    #     y : numpy array
    #         Array with y-values. At every time point a prediction is made (excluding the current y)
    #     x : numpy array or None
    #         If prediciton requires inputs, provide them here
    #     steps_ahead : int (default 1)
    #         The number of steps to predict into the future at every time point
    #     y_start : float or None
    #         Starting value from which to start prediction. If None, last stored value from training will be used
    #     Returns
    #     -------
    #     y_predicted : numpy array
    #         Array of predictions at every time step of shape (times, steps_ahead)
    #     """

    #     # Check if ESN has been trained
    #     if self.out_weights is None or self.y_last is None:
    #         raise ValueError('Error: ESN not trained yet')

    #     # Normalize the arguments (like was done in train)
    #     y = self.scale(outputs=y)
    #     if not x is None:
    #         x = self.scale(inputs=x)

    #     # Timesteps in y
    #     t_steps = y.shape[0]

    #     # Check input
    #     if not x is None and not x.shape[0] == t_steps:
    #         raise ValueError('x has the wrong size for prediction: x.shape[0] = {}, while y.shape[0] = {}'.format(
    #             x.shape[0], t_steps))

    #     # Choose correct input
    #     if x is None and not self.feedback:
    #         #pass #raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
    #         inputs = torch.ones((t_steps + steps_ahead, 2), **dev) 
    #     elif not x is None:
    #         # Initialize input
    #         inputs = torch.ones((t_steps, 1), **dev)  # Add bias term
    #         inputs = torch.hstack((inputs, x))  # Add x inputs
    #     else:
    #         # x is None
    #         inputs = torch.ones((t_steps + steps_ahead, 1), **dev)  # Add bias term
        
    #     # Run until we have no further inputs
    #     time_length = t_steps if x is None else t_steps - steps_ahead + 1

    #     # Set parameters
    #     y_predicted = torch.zeros((time_length, steps_ahead), dtype=self.dtype, device=self.device)

    #     # Get last states
    #     previous_y = self.y_last
    #     if not y_start is None:
    #         previous_y = self.scale(outputs=y_start)[0]

    #     # Initialize state from last availble in train
    #     current_state = self.state[-1]

    #     # Predict iteratively
    #     with torch.no_grad():
            
    #         for t in range(time_length):

    #             # State_buffer for steps ahead prediction
    #             prediction_state = current_state.clone().detach()
                
    #             # Y buffer for step ahead prediction
    #             prediction_y = previous_y.clone().detach()
            
    #             # Predict stepwise at from current time step
    #             for n in range(steps_ahead):
                    
    #                 # Get correct input based on feedback setting
    #                 prediction_input = inputs[t + n] if not self.feedback else torch.hstack((inputs[t + n], prediction_y))
                    
    #                 # Update
    #                 prediction_update = self.activation_function(torch.matmul(self.in_weights, prediction_input.T) + 
    #                                                torch.matmul(self.weights, prediction_state))
                    
    #                 prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state
                    
    #                 # Store for next iteration of t (evolves true state)
    #                 if n == 0:
    #                     current_state = prediction_state.clone().detach()
                    
    #                 # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
    #                 prediction_row = torch.hstack((prediction_input, prediction_state))
    #                 if not self.backprop:
    #                     y_predicted[t, n] = torch.matmul(prediction_row, self.out_weights)
    #                 else:
    #                     y_predicted[t, n] = self.LinOut.weight.T @ prediction_row[1:]
    #                 prediction_y = y_predicted[t, n]

    #             # Evolve true state
    #             previous_y = y[t]

    #     # Denormalize predictions
    #     y_predicted = self.descale(outputs=y_predicted)
        
    #     # Return predictions
    #     return y_predicted

#################################################
    #RcNetwork vestigal method:
    # def back(self, tensor_spec, retain_graph = True):
    #     return tensor_spec.backward(torch.ones(*tensor_spec.shape, device = tensor_spec.device), retain_graph = retain_graph)

#################################################

# def execute_backprop(RC):

#     gd_weights = []
#     gd_biases = []
#     ys = []
#     ydots =[]
#     scores = []
#     Ls = []
#     init_conds_clone = init_conditions.copy()
#     if not SOLVE:
#         orig_weights = self.LinOut.weight.clone()
#         orig_bias = self.LinOut.bias.clone()
        
#     for i, y0 in enumerate(init_conds_clone[0]):
#         #print("w", i)
#         if SOLVE:
#             self.LinOut.weight = Parameter(self.weights_list[i].view(self.n_outputs, -1)).requires_grad_(True)
#             self.LinOut.bias = Parameter(self.biases_list[i].view(1, self.n_outputs)).requires_grad_(True)
#         else:
#             self.LinOut.weight = Parameter(orig_weights.view(self.n_outputs, -1))
#             self.LinOut.bias = Parameter(orig_bias.view(1, self.n_outputs))
#         self.init_conds[0] = float(y0)
#         #print(self.init_conds[0])
#         with torch.enable_grad():
#             weight_dict = backprop_f(self, force_t = self.force_t, custom_loss = ODE_criterion, epochs = epochs)

#         score=weight_dict["best_score"]
#         y = weight_dict["y"]
#         ydot = weight_dict["ydot"]
#         loss, gd_weight, gd_bias = weight_dict["loss"]["loss_history"], weight_dict["weights"],  weight_dict["bias"]
#         scores.append(score)
#         ys.append(y)
#         ydots.append(ydot)
#         gd_weights.append(gd_weight)
#         gd_biases.append(gd_bias)
#         Ls.append(loss)

############################# More RcNetwork vestigal methods ###################################

# def plot_reservoir(self):
    #     """Plot the network weights"""
    #     sns.histplot(self.weights.cpu().numpy().view(-1,))

    # def forward(self, t, input_, current_state, output_pattern):
    #     """
    #     Arguments:
    #         t: the current timestep
    #         input_: the input vector for timestep t
    #         current_state: the current hidden state at timestep t
    #         output_pattern: the output pattern at timestep t.
    #     Returns:
    #         next_state: a torch.tensor which is the next hidden state
    #     """
    #     # generator = self.random_state, device = self.device)

    #     preactivation = self.LinIn(input_) + self.bias_ + self.LinRes(current_state)

    #     if self.feedback:
    #         preactivation += self.LinFeedback(output_pattern)
        
    #     #alternative: uniform noise
    #     #self.noise = torch.rand(self.n_nodes, **self.tensor_args).view(-1,1) if noise else None

    #     update = self.activation_function(preactivation) # + self.PyESNnoise * (self.external_noise - 0.5)
    #     if self.noise != None:
    #         #noise_vec = torch.normal(mean = torch.zeros(self.n_nodes, device = self.device), 
    #         #                              std = torch.ones(self.n_nodes, device = self.device),
    #         #                              generator = self.random_state)* self.noise
    #         noise_vec = torch.rand(self.n_nodes, **self.tensor_args) * self.noise
    #         update += noise_vec 
    #     next_state = self.leaking_rate[0] * update + self.leaking_rate[1] * current_state
    #     return next_state

    # def preactivation_beta(self, t, input_vector, recurrent_vec, bias, betas):
    #     return input_vector + recurrent_vec +  self.bias * self.beta[t-1,:]


# def activate(self, dt):
    #     alpha = self.alpha ** dt, self.alpha ** (1 - dt)

###################### This function was defined in the error method #################

#### attempt at loss function when steps ahead > 2 

        # def step_ahead_loss(y, yhat, plot = False, decay = 0.9):
        #     loss = torch.zeros(1,1, device = self.device)
        #     losses = []
        #     total_length = len(y)
        #     for i in range(1, total_length - self.steps_ahead):
        #         #step ahead == i subsequences
        #         #columnwise
        #         #   yhat_sub = yhat[:(total_length - i), i - 1]
        #         #   y_sub = y[i:(total_length),0]
        #         #row-wise
        #         yhat_sub = yhat[i-1, :]
        #         y_sub = y[i:(self.steps_ahead + i),0]
        #         assert(len(yhat_sub) == len(y_sub)), "yhat: {}, y: {}".format(yhat_sub.shape, y_sub.shape)

        #         loss_ = nmse(y_sub.squeeze(), yhat_sub.squeeze())

        #         if decay:
        #             loss_ *= (decay ** i)

        #         #if i > self.burn_in:
        #         loss += loss_
        #         losses.append(loss_)

        #     if plot:
        #         plt.plot(range(1, len(losses) + 1), losses)
        #         plt.title("loss vs step ahead")
        #         plt.xlabel("steps ahead")
        #         plt.ylabel("avg loss")
        #     return loss.squeeze()

        # if predicted.shape[1] != 1:
        #     return step_ahead_loss(y = target, yhat = predicted) 


# def fit_unsupervised(self):
    #     """
    #     if the esn is unsupervised, this fork.
    #     """
    #     pass

    # def fit_hamiltonian(self):
    #     """
    #     for the new project
    #     """


# def calculate_n_grads(self, X, y,  n = 2, scale = False):
    #     self.grads = []

    #     #X = X.reshape(-1, self.n_inputs)

    #     assert y.requires_grad, "entended doesn't require grad, but you want to track_in_grad"
    #     for i in range(n):
    #         print('calculating derivative', i+1)
    #         if not i:
    #             grad = _dfx(X, y)
    #         else:
    #             grad = _dfx(X, self.grads[i-1])

    #         self.grads.append(grad)

    #         if scale:
    #             self.grads[i] = self.grads[i]/(self._input_stds)
    #     with torch.no_grad():
    #         self.grads = [self.grads[i][self.burn_in:] for i in range(n)]
                
    #         #self.yfit = self.yfit[self.burn_in:]
    #     #assert extended_states.requires_grad, "entended doesn't require grad, but you want to track_in_grad"


 ################# center_H ################## Failed idea, unclear whether that's my fault (likely) or Hargun's (unlikely)

 # def _center_H(self, inputs = None, outputs = None, keep : bool = False):
 #        """
 #        Centers the hidden states?

 #        INSTRUCTIONS:
 #        1. assign `_x_means` to self, along the axis such that 
 #           the numbers of means matches the number of features (2)
 #        2. assign `_y_mean` to self (y.mean())
 #        3. subtract _x_means from X and assign it to X_centered
 #        4. subtract _y_mean from y and assign it to y_centered

 #        Parameters
 #        ----------
 #        inputs : tensor or array-like
 #            Description of arg1
 #        outputs : tensor or array-like
 #            Description of arg2
 #        keep : bool
 #            if true the means and standard deviations will be saved

 #        Returns
 #        -------
 #        int
 #            Description of return value

 #        """
 #        if inputs is not None:
 #            X = inputs

 #            if keep:
 #                self._x_means = X.mean(axis=0)
 #                self._x_stds = X.std(axis = 0)

 #            X_centered = (X - self._x_means)/self._x_stds
 #            return X_centered
 #        if outputs is not None:
 #            y = outputs

 #            if keep:
 #                self._y_means = y.mean(axis = 0)

 #            y_centered = y - self._y_means #(y - y_means)/y_stds

 #            return y_centered


 ##################

 # in_weights = torch.rand(n, m, generator = self.random_state, device = self.device, requires_grad = False)
    #                 in_weights =  (in_weights * 2) - 1
    #                 if self.input_connectivity is not None:
    #                     accept = torch.rand(n, m, **self.tensor_args) < self.input_connectivity
    #                     in_weights *= accept


  ##############

  #def _check_device_cpu(self):
    #    #TODO: make a function that checks if a function is on the cpu and moves it there if not
    #    pass