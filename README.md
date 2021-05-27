RcTorch
=========
A Pytorch toolset for creating and optimizing Echo State Networks. 

>License: 2020-2021 MIT  
>Authors: Hayden Joy, Marios Mattheakis

Contains:
- A ESN Reservoir architecture class "esn.py"
- Bayesian Optimization (BO) class "esn_cv.py" with optimized routines for Echo State Nets through `Botorch` (GPU optimized), can train multiple RCs in parellel durring BO
  - an implimentation of the TURBO-1 algorithm as outlined in this paper: https://github.com/uber-research/TuRBO
- Capable of solving differential equations (the population equation, the bernoulli equation, a simple harmonic oscillator and a nonlinear oscillator)

Reference to prior instantiation:  
This library is an extension and expansion of a previous library written by Reinier Maat: https://github.com/1Reinier/Reservoir
2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-7. IEEE, 2018  
https://arxiv.org/abs/1903.05071

## For example usage please see the notebooks folder.