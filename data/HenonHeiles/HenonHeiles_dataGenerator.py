#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:50:04 2020

@author: marios
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Use below in the Scipy Solver   
def f(u, t ,lam=1):
    x, y, px, py = u      # unpack current values of u
    derivs = [px, py, -x -2*lam*x*y, -y -lam*(x**2-y**2) ]     # list of dy/dt=f functions
    return derivs

# Scipy Solver   
def HHsolution(N,t, x0, y0, px0, py0,lam=1):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f, u0, t, args=(lam,))
    xP = solPend[:,0];    yP  = solPend[:,1];
    pxP = solPend[:,2];   pyP = solPend[:,3]
    return xP,yP, pxP, pyP


# Set the initial state. lam controls the nonlinearity
x0, y0, px0, py0, lam =  0.3,-0.3, 0.3, 0.15, 1; 
t0, t_max, N = 0.,100*np.pi, 1000; 

dt = t_max/N; 
X0 = [t0, x0, y0, px0, py0, lam]
t_num = np.linspace(t0, t_max, N)

# E0, E_ex = HH_exact(N,x0, y0, px0, py0, lam)
x_num, y_num, px_num, py_num = HHsolution(N,t_num, x0, y0, px0, py0, lam)


np.savetxt("t.dat",t_num)
np.savetxt("x.dat",x_num)
np.savetxt("y.dat",y_num)

#np.savetxt("t.txt",t_num)
#%np.savetxt("x.txt",x_num)
#np.savetxt("y.txt",y_num)

#np.savetxt("px.txt",px_num)
#np.savetxt("py.txt",py_num)
