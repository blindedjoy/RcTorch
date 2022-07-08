
Exceptions
==========

.. autofunction:: RcNetwork(n_nodes = 1000, bias = 0, connectivity = 0.1, leaking_rate = 0.99, spectral_radius = 0.9, noise = None, #<-- important hyper-parameters
                 regularization = None, feedback = False, enet_alpha = None, gamma_cyclic = None,
                 mu = None, sigma = None,                     #<-- activation, feedback
                 input_scaling = 0.5, feedback_scaling = 0.5,                                    #<-- hyper-params not needed for the hw
                 approximate_reservoir = False, device = None, id_ = None, random_state = 123, reservoir = None, #<-- process args
                 classification = False, l2_prop = 1, n_inputs = None, n_outputs = None,
                  dtype = None, calculate_state_grads = False, dt = None,
                 activation_function = "tanh", enet_strength = None,
                 output_activation = "identity", input_weight_dist = "uniform", input_connectivity = None,
                 #act_f_prime = sech2,
                 gamma = None, spikethreshold = None, reservoir_weight_dist = "uniform", solve_sample_prop = 1,
                 feedback_weight_dist = "uniform", feedback_connectivity = None,
                 **kwargs)
Functions and Classes
===============================

.. automodule:: rctorchprivate.rc_bayes
   :members:

    .. autoclass:: TurboState

.. automodule:: rctorchprivate

