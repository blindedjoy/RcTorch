else:
            #+++++++++++++++++++++++++++++++         backprop          +++++++++++++++++++++++++++++++
            trainable_parameters = []
            trainable_parameters = []

            for p in self.parameters():
                if p.requires_grad:
                    trainable_parameters.append(p)

            for i, p in enumerate(trainable_parameters):
                print(f'Trainable parameter {i} {p.name} {p.data.shape}')


            running_loss = 0
            train_losses = []
            if not optimizer:
                optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            min_loss = float("Inf")
            epochs_not_improved = False

            bias_buffer = torch.ones((y.shape[0],1),**self.dev)

            
            if self.epochs == None:
                endless = True
            else:
                endless = False

            self.freeze_weights()
            assert self.LinOut.weight.requires_grad

            states = self.state.clone()
            states = states.to(self.device)


            ######.  TRAIN STATES ################### SEPARATELY FROM THE OTHER FORWARD PASS
            #X_detached = self.X.detach()
            #X_detached.requires_grad_(True)

            
                #self.dh_dx = torch.zeros(0,**self.dev)
            X_detached = self.unscaled_X.clone().detach().requires_grad_(True)
            if scale_x:
                X_detached = (X_detached- self._input_means) / self._input_stds
            if out_weights:
                self.LinOut.weight.data = out_weights["weight"]
                self.LinOut.bias.data = out_weights["bias"]
                #self.LinOut.weight.requires_grad_(False)
                #self.LinOut.bias.requires_grad_(False)
            state_list = []

            if self.feedback:
                assert False, "not implimented"

            """

            for t in range(0, X_detached.shape[0]):
                input_t = X_detached[t, :].T
                state_t, output_t = self.train_state(t, X = input_t,
                                          state = states[t,:], 
                                          y = None, output = True)
                state_list.append(state_t)
                with no_grad():
                    dht_dt = dfx(X_detached, state_t)
                    dht_dy = dfx(state_t, output_t)
                    if t > 2:
                            dht_dh = dfx(state_list[t-1], state_t)
                    dyt_dx = dfx(X_detached, output_t) #<-- calculating the derivative at each timestep.
                if t == 0:
                    self.dh_dhs = []
                    self.dh_dts = []
                    self.dh_dys = []
                    self.dy_dxs = []
                    self.outputs = []
                    self.outputs = []
                    self.dh_dt = dht_dt
                    #self.dy_dh = 
                    #elif t > 1:
                    self.dh_dt = torch.cat((self.dh_dt, dht_dt))
                    #self.dy_dh = torch.cat((self.dh_dx, dht_dx))
                
                self.dh_dys.append(dht_dy)
                self.dh_dts.append(dht_dt)
                self.outputs.append(output_t)
                self.dy_dxs.append(dyt_dx)
                self.dh_dhs.append(dyt_dx)

                states = cat([states, state_t.view(-1, self.n_nodes)], axis = 0)
            #####################################################
            self.dy_dx = dfx(X_detached, torch.vstack(self.outputs)) #one time derivative doesn't work very well.
            #####################################################################
            
            del states


            self.hidden_transitions = torch.hstack(self.dh_dhs).cpu().T
            self.dy_dx_matrix = torch.hstack(self.dy_dxs)
            #del self.dy_dxs
            #proper scaling:
            self.dy_dx = self.dy_dx_matrix.sum(axis = 0 )# / self._input_stds)*self._output_stds
            #undo the output layer:
            self.dh_dx =  self.dy_dx / (self.LinOut.weight.view(-1,self.n_outputs) * (self.n_nodes + 1))
            """

            #self.dh_dx_for_backwards_pass = self.dh_dx.mean(axis = 0).view(-1,1).clone().detach()
            #assert False

            #extended_states = hstack((X_detached, states[1:]))

            self.loss_history, reassign = [], False
            base_derivative_calculated = False   

            
            
            # The recurrence class will do the forward pass for backprop. We will give it the derivative after doing another forward pass
            
            #order: ctx, input,  states, LinIn, LinOut, LinRes, bias_, n_nodes, activation_function, leaking_rate, noise, feedback, y,  tensor_args, LinFeedback = None
            
            
            final_grad = None
            if 1 == 0:
                pass
            else:
                for e in range(self.epochs):
                    self.states = self.state.clone()
                    self.states = self.states.to(self.device).view(1,-1)

                    optimizer.zero_grad()
                    """
                    for t in range(0, X_detached.shape[0]):
                        input_t = self.X[t, :].T
                        train_states_class = Recurrence()
                        state_t = train_states_class.apply(input_t, 
                                                           self.states[t,:],
                                                           self.LinIn, 
                                                           self.LinOut, 
                                                           self.LinRes, 
                                                           self.bias, 
                                                           self.n_nodes, 
                                                           self.activation_function, 
                                                           self.leaking_rate, 
                                                           self.noise, 
                                                           self.feedback, 
                                                           y, 
                                                           self.tensor_args,
                                                           self.dh_dts[t].detach(),
                                                           self.hidden_transitions[t,:].detach(),
                                                           None)
                        """
                    for t in range(0, self.X.shape[0]):
                        input_t = self.X[t, :].T
                        state_t, output_t  = self.train_state(t, X = input_t,
                                          state = self.states[t,:], 
                                          y = None, output = True, retain_grad = True)

                        state_t.retain_grad()

                        if full_grads:
                            dyt_dx = dfx(self.X, output_t) 
                            if not t:
                                self.dy_dxs = [dyt_dx]
                            else:
                                self.dy_dxs.append(dyt_dx)


                        self.states=  cat([self.states, state_t.view(-1, self.n_nodes)], axis = 0)
                        if not t:
                            outputs = output_t
                        else:
                            outputs = torch.cat((outputs, output_t), axis = 0)

                    if full_grads:
                        self.dy_dx_matrix = torch.hstack(self.dy_dxs)
                        self.dy_dx = self.dy_dx_matrix.sum(axis = 0 )

                    extended_states = hstack((self.X.detach(), self.states[1:]))
                    self.yfit = self.forward(extended_states)

                    
                    
                    #self.dy_dh = dfx(states, self.yfit)
                    #with torch.no_grad():
                    #    if self.track_in_grad or ODE:
                    #        self.dy_dx_orig = dfx(self.X, self.yfit) 
                    #        self.dh_dx = self.dy_dx_orig / (self.LinOut.weight * self.LinOut.weight.shape[1])
                    if ODE or self.track_in_grad:
                        #assert self.yfit.shape == self.dh_dx.shape, f'{self.yfit.shape} != {self.dh_dx.shape}'
                        if scale_x:
                            yfit = self.yfit  / self._input_stds
                        with torch.no_grad():
                            if not full_grads:
                                self.dy_dx = dfx(self.X, self.yfit)
                            else:
                                self.dy_dx = self.dy_dx / self._input_stds
                    if ODE:
                        loss = self.criterion(self.X, self.yfit, self.dy_dx)
                    else:
                        #self.yfit = self.yfit * self._output_stds + self._output_means
                        loss = self.criterion(self.yfit, y)

                    assert loss.requires_grad
                    #assert False, loss
                    loss.backward(retain_graph = True)
                    assert type(self.X.grad != None)

                    #save best weights
                    if e > save_after_n_epochs:
                        
                        if float(loss) < min(self.loss_history):
                            best_bias, best_weight = self.LinOut.bias.clone(), self.LinOut.weight.clone()
                            self.LinOut.bias.data, self.LinOut.weight.data = best_bias, best_weight.view(*self.LinOut.weight.shape)
                            self.final_grad = self.dy_dx.clone()

                    self.loss_history.append(float(loss))

                    optimizer.step()
                    if e % 100 == 0:
                        print("Epoch: {}/{}.. ".format(e+1, self.epochs),
                              "Training  Loss: {:.3f}.. ".format(torch.log(loss)))

                    #early stopping
                    if self.patience:
                        if e > 10:
                            if loss < min_loss:
                                epochs_not_improved = 0
                                min_loss = loss
                            else:
                                epochs_not_improved += 1

                        if e > 10 and epochs_not_improved >= self.patience:
                            print('Early stopping at epoch' ,  e, 'loss', loss)
                            early_stop = True
                            
                            break
                            
                        else:
                            continue
                    e=e+1

                    #to avoid unstable solutions consider an additional convergence parameter


                #early stopping code from the following article:
                #https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
            
            #for the final state we want to save that in self.state
            self.out_weights = self.LinOut.weight
            self.out_weights._name_ = "out_weights"
            #extended_states = hstack((self.X, self.state))