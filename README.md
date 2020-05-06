# Fitting Models to Financial Time Series
#### 4th Year Project Code Repository
## Introduction
This repository contains all of the code used throughout the project. 
This includes code which was used as part of the learning and exploration parts of the project. As a result some code will not
have been directly used to generate the results in the final report.  
## Installation
The code in this repository can be used *as is* in conjunction with some other python packages:
- Numpy
- Matplotlib
- Pandas 
- Scipy

## Features

### Neural Network
The neural network uses a class `NeuralNet` to contain the creation, training and prediction for a basic 3 layer neural
network. The network is trained using backpropagation and gradient descent. The activation functions are defined in `utilities.py`.
For example, to create and train a basic three layer 48-64-1 network:
```python
from basic_neural_network import NeuralNet

# Specify the path to the data set
path = "Data Sets\\Electricity\\UK Elec Hourly - no weekends.csv"

# Create a network class
network = NeuralNet(path, input_size=48, hidden_size=64, do_sub_sample=False)

# Train the network
network.train_network(epochs=100)
```

### Independent Component Analysis
The `ica.py` file provides many functions to assist in computing an independent component analysis. For example,
with a mixed time series:
```python
from ica import whiten_data, comp_ica
from utilities import normalise

# Compute centred data
normalised_data = normalise(data)

# Compute whitened data
whitened_data = whiten_data(normalised_data)

# Compute independent components
independent_components, unmixing_matrix = comp_ica(normalised_data)
```

### Particle Filtering
To learn the parameters of a stochastic volatility model the ```ParticleFilter``` or ```ParticleFilterBackTrace``` classes
can be used. The latter uses a more memory efficient implementation although it will be slower for short data sets 
(<500 points). 
There are a few different instances of this class which all inherit standard features but allow for variation in 
the model assumptions. There are also
two different backplanes; ```ParticleFilter``` and ```ParticleFilterBackTrace```. 
```python
from particle_filter_standard import ParticleFilterStandard

# Create data from a source
data_observed, data_hidden = generate_data()

# Create a particle filter object
particle_filter = ParticleFilterStandard(data_observed,
                                         num_particles=100,
                                         a=0.5,
                                         b=0.5,
                                         c=0.5,
                                         true_hidden=data_hidden,
                                         num_iterations=100,
                                         learn_rate=0.001,
                                         do_adaptive_learn=True
                                         )

# Do one pass of the filter and plot the results
particle_filter.filter_pass()
particle_filter.plot_filter_pass()

# Calibrate the model and show the parameter evolution
particle_filter.calibrate_model()
particle_filter.plot_params()
```


### Data Sets
The code requires data sets to do work on. There are two main methods for generating them; synthetic and real data.  

#### Loading Real Data
They should be provided in a `.csv` format
with the dates in the first column of the data. A function is provided for importing from `.csv` files; `utilites.load_data`.
When using this function the dates will be inner joined
such that only the dates (specifically standardised times) found in every specified data set will be used.

#### Generating Synthetic data
For testing purposes or model validation it may be useful to create synthetic data generated from a range of stochastic
volatility models. This functionality is provided by `stochastic_volatility.py` which contains a number
of volatility models:
- `gen_univ_sto_vol` - generates from the univariate stochastic volatility model
- `gen_multi_sto_vol`- generates from the multivariate stochastic volatility model 
- `gen_univ_mrkv` - generates from a general hidden Markov model with Gaussian innovations and observations
- `gen_univ_gamma` - generates from a stochastic volatility model with gamma distributed observations

### Plotting
The file `plot_utils.py` offers a large range of plotting utility functions. Most functions simply provide a wrapper 
to the standard `matplotlib` functions. In addition, there are a few special purpose functions which make plotting
multi-dimension series into subplots simpler.

