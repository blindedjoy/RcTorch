Reservoir
=========
A Python 3 toolset for creating and optimizing Echo State Networks.

>Author: Reinier Maat, Nikos Gianniotis  
>License: MIT  
>2016-2019  

Contains:
- Vanilla ESN and Simple Cyclic Reservoir architectures.
- Bayesian Optimization with optimized routines for Echo State Nets through `GPy`.
- Clustering routines to cluister time series by optimized model.

Reference:  
2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-7. IEEE, 2018  
https://arxiv.org/abs/1903.05071

## Example Use
```python
# Load data
data = np.loadtxt('example_data/MackeyGlass_t17.txt')
train = data[:4000].reshape(-1, 1)
test = data[4000:4100].reshape(-1, 1)

# Set optimization bounds
bounds = {'input_scaling': (0, 1),
          'feedback_scaling': (0, 1),
          'leaking_rate': (0, 1),
          'spectral_radius': (0, 1.25),
          'regularization': (-12, 1),
          'connectivity': (-3, 0),
          'n_nodes': (100, 1500)}

# Set optimization parameters
esn_cv = EchoStateNetworkCV(bounds=bounds,
                            initial_samples=100,
                            subsequence_length=1000,
                            eps=1e-3,
                            cv_samples=1,
                            max_iterations=1000,
                            scoring_method='tanh',
                            verbose=True)

# Optimize
best_arguments = esn_cv.optimize(y=train)

# Build best model
esn = EchoStateNetwork(**best_arguments)
esn.train(y=train)
score = esn.test(y=test, scoring_method='nrmse')

# Diagnostic plot
plt.plot(esn.predict(100), label='Predicted')
plt.plot(test, label='Ground truth')
plt.title('Prediction on next 100 steps')
plt.legend()
plt.show()

# Print the score
print("Test score found:", score)

```

