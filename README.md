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

## Examples

#### Neural Network
To create and train a basic three layer 48-64-1 network:
```python
from basic_neural_network import NeuralNet

# Specify the path to the data set
path = "Data Sets\\Electricity\\UK Elec Hourly - no weekends.csv"

# Create a network class
network = NeuralNet(path, input_size=48, hidden_size=64, do_sub_sample=False)

# Train the network
network.train_network(epochs=100)
```

#### Independent Component Analysis
To perform independent Component Analysis on a mixed time series:
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

#### Particle Filtering
To learn the parameters of a stochastic volatility model the ParticleFilter class can be used. There are a few different 
versions of this class which all inherit standard features but allow for variation 
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
The code requires data sets to do work on. There are two main methods for generating it; synthetic and real data.  

#### Loading Real Data
They should be provided in a `.csv` format
with the dates in the first column of the data. A function is provided for importing from `.csv` files; `utilites.load_data`.
When using this function the dates will be inner joined
such that only the dates (specifically standardised times) found in every specified data set will be used.

#### Generating Synthetic data
For testing purposes or model validation it may be useful to create synthetic data generated from a range of stochastic
volatility models. This functionality is provided by `stochastic_volatility.py` which contains a number
of volatility models:
- `gen_univ`
- `gen_mrkv`
