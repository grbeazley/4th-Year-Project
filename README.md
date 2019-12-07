# Fitting Models to Financial Time Series
#### 4th Year Project Repository
## Introduction
This repository contains all of the code used throughout the project. 
This includes code which was used as part of the learning process and not in any outputs in the report. 
## Installation
The code in this repository can be used *as is* in conjunction with some other python packages:
- Numpy
- Matplotlib
- Pandas 

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
```pythonstub
from ica import whiten_data, comp_ica
from utilities import normalise

# Compute centred data
normalised_data = normalise(data)

# Compute whitened data
whitened_data = whiten_data(normalised_data)

# Compute independent components
independent_components, mixing_matrix = comp_ica(normalised_data)
```

### Data Sets
The code requires data sets to do work on. They should be provided in a `.csv` format
with the dates in the first column of the data. When using the `utilites.load_data` the dates will be inner joined
such that only the dates (specifically standardised times) found in every specified data set will be used. 
