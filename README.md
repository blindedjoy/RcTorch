<img src="https://github.com/blindedjoy/RcTorch/blob/master/fig/png/rctorch_logo.png?raw=true" alt="drawing" style="width:220px;"/>

## Package details and supporting resources

RcTorch is a Pytorch toolset for creating and optimizing Reservoir Computers (RCs). 
See the complete RcTorch documentation at [readthedocs](https://rctorch.readthedocs.io/en/latest/).
Also see the preprint of our research paper on the [Arkiv](https://arxiv.org/abs/2207.05870) which demonstrates the power of our software by solving the forced pendulum.

>License: 2020-2022 MIT  
>Authors: Hayden Joy, Marios Mattheakis (hnjoy@mac.com, mariosmat@seas.harvard.edu)




Contains:
- An RC architecture class "rc.py"
- Bayesian Optimization (BO) class "rc_bayes.py" with optimized routines for RC neural networks through `Botorch` (GPU optimized), can train multiple RCs in parallel during BO
  - an implimentation of the TURBO algorithm as outlined in this paper: https://github.com/uber-research/TuRBO
- Capable of solving differential equations (the population equation, the bernoulli equation, a simple harmonic oscillator and a nonlinear oscillator)

Reference to prior instantiation:  
This library is an extension and expansion of a previous library written by Reinier Maat: https://github.com/1Reinier/Reservoir
2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-7. IEEE, 2018  
https://arxiv.org/abs/1903.05071

## For example usage please see the notebooks folder.

# Installation

## Using pip

Like most standard libraries, `rctorch` is hosted on [PyPI](https://pypi.org/project/RcTorch/). To install the latest stable release, 

```bash
pip install -U rctorch  # '-U' means update to latest version
```
## Example Usages

### Imports

```python
from rctorch import *
import torch
```

### Load data

RcTorch has several built in datasets. Among these is the forced pendulum dataset. Here we demonstrate
```python
fp_data = rctorch.data.load("forced_pendulum", train_proportion = 0.2)

force_train, force_test = fp_data["force"]
target_train, target_test = fp_data["target"]

#Alternatively you can use sklearn's train_test_split.
```

### Hyper-parameters


```python
#declare the hyper-parameters
>>> hps = {'connectivity': 0.4,
           'spectral_radius': 1.13,
           'n_nodes': 202,
           'regularization': 1.69,
           'leaking_rate': 0.0098085,
           'bias': 0.49}
```

### Setting up your very own EchoStateNetwork (pure prediction)

```python
my_rc = RcNetwork(**hps, random_state = 210, feedback = True)

#fitting the data:
my_rc.fit(y = target_train)

#making our prediction
score, prediction = my_rc.test(y = target_test)
my_rc.combined_plot()

```

![](https://github.com/blindedjoy/RcTorch/blob/master/fig/png/traj_1.png?raw=true)

Top plot: Above the ground truth data of the forced pendulum is plotted as dashed lines. The position training set is plotted in yellow and the position prediction is plotted in red. The momentum training set prediction is plotted in blue and the test set prediction is plotted in red.

Bottom plot: The mean squared error plot. The colors correspond to the plot above. For more information see our [Arkiv paper](https://arxiv.org/abs/2207.05870). 


Note: Feedback allows the network to feed in the prediction at the previous timestep as an input. This helps the RC to make longer and more stable predictions in many situations.

### Setting up your very own EchoStateNetwork (parameter-aware version)

In order to add observers (see this [paper](https://aip.scitation.org/doi/abs/10.1063/1.4979665), simply add X as an argument for the RC. In effect this allows the RC to take the force applied as input (this would be known in a situation where a robotic arm were programmed to push on a pendulum for example) and then learn the mapping from the force to the target (in this case the position and momentum of a pendulum).

```python
my_rc = RcNetwork(**hps, random_state = 210, feedback = True)

#fitting the data:
my_rc.fit(X = force_train, y = target_train)

#making our prediction
score, prediction = my_rc.test(X = force_test, y = target_test)
my_rc.combined_plot()

```

![](https://github.com/blindedjoy/RcTorch/blob/master/fig/png/traj_2.png?raw=true)

### Bayesian Optimization

Unlike most other reservoir neural network packages RcTorch is capable of automatically tune hyper-parameters, saving researchers time and energy. In addition RcTorch predictions are world class!

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

![](https://github.com/blindedjoy/RcTorch/blob/master/fig/png/BO.png?raw=true)

