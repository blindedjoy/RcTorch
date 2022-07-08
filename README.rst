====================
RcTorch Readme file
====================

A Pytorch toolset for creating and optimizing Echo State Networks.

    - License: 2020-2022 MIT  
    - Authors: Hayden Joy, Marios Mattheakis

.. contents::


Contains:
- A ESN Reservoir architecture class "rc.py"
- Bayesian Optimization (BO) class "rc_bayes.py" with optimized routines for Echo State Nets through `Botorch` (GPU optimized), can train multiple RCs in parellel durring BO
  - an implimentation of the TURBO-1 algorithm as outlined in this paper: https://github.com/uber-research/TuRBO
- Capable of solving differential equations (the population equation, the bernoulli equation, a simple harmonic oscillator and a nonlinear oscillator)



Example Usage
=============


Installation
------------

Like most standard libraries, `rctorch` is hosted on [PyPI](https://pypi.org/project/RcTorch/). To install the latest stable relesase, 

```bash
pip install -U rctorch  # '-U' means update to latest version
```


Importing RcTorch
-----------------

```python
from rctorch import *
import torch
```

Load data
---------

RcTorch has several built in datasets. Among these is the forced pendulum dataset. Here we demonstrate [...]

```python
fp_data = rctorch.data.load("forced_pendulum", train_proportion = 0.2)

force_train, force_test = fp_data["force"]
target_train, target_test = fp_data["target"]

#Alternatively you can use sklearn's train_test_split.
```

Hyper-parameters
----------------


```python
#declare the hyper-parameters
>>> hps = {'connectivity': 0.4,
           'spectral_radius': 1.13,
           'n_nodes': 202,
           'regularization': 1.69,
           'leaking_rate': 0.0098085,
           'bias': 0.49}
```

Setting up your very own EchoStateNetwork
-----------------------------------------

```python
my_rc = RcNetwork(**hps, random_state = 210, feedback = True)

#fitting the data:
my_rc.fit(y = target_train)

#making our prediction
score, prediction = my_rc.test(y = target_test)
my_rc.combined_plot()

```

![](https://raw.githubusercontent.com/blindedjoy/RcTorch-private/blob/master/resources/pure_prediction1.jpg)



Feedback allows the network to feed in the prediction at the previous timestep as an input. This helps the RC to make longer and more stable predictions in many situations.


Bayesian Optimization
---------------------

Unlike most other reservoir neural network packages ours offers the automatically tune hyper-parameters.

```python

#any hyper parameter can have 'log_' in front of it's name. RcTorch will interpret this properly. 
bounds_dict = {"log_connectivity" : (-2.5, -0.1), 
               "spectral_radius" : (0.1, 3),
               "n_nodes" : (300,302),
               "log_regularization" : (-3, 1),
               "leaking_rate" : (0, 0.2),
               "bias": (-1,1),
               }
rc_specs = {"feedback" : True,
             "reservoir_weight_dist" : "uniform",
             "output_activation" : "tanh",
             "random_seed" : 209}

rc_bo = RcBayesOpt(bounds = bounds_dict, 
                    scoring_method = "nmse",
                    n_jobs = 1,
                    cv_samples = 3,
                    initial_samples= 25,
                    **rc_specs
                    )
```

Special Thanks
==============

This library is an extension and expansion of a previous library written by Reinier Maat.

  `Github link (Reservoir) <https://github.com/1Reinier/Reservoir>`_

  `Efficient Optimization of Echo State Networks for Time Series Datasets:  <https://arxiv.org/abs/1903.05071>`_

  2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-7. IEEE, 2018  

  
