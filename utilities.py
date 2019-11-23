import numpy as np
import pandas as pd

def sigmoid(x):
    """
    Computes the sigmoid function for scalar or vector x
    """
    return 1 / (1 + np.exp(-x))


def sym_sigmoid(x):
    """
    Computes symmetric sigmoid function based on tanh
    Works for scalar or vector input x
    y = a*tanh(s*x)
    y = a - 2a / ( 1 + exp(2*s*x) )
    """
    a = 1.7159
    s = 2/3
    return a - 2*a / (1 + np.exp(2*s*x))


def dev_sym_sigmoid(x):
    """
    Computes the derivative of the symmetrical sigmoid activation function
    """
    a = 1.7159
    s = 2/3
    exponential = np.exp(2*s*x)
    return 4*a*s*exponential / (1 + exponential) ** 2


def softmax(x):
    # Returns softmax of an array of exponents
    exps = np.exp(x)
    return exps / exps.sum()


def log_returns(time_series):
    # Computes the log returns for an input time series
    # Assumes in chronological order and takes Pt - Pt-1
    return np.log(time_series[:-1]) - np.log(time_series[1:])


def load_data(stem, paths_dict, index_type='str'):
    # Loads all csvs based on stem and paths dictionary
    # Assumes first path has correct data length (by dates)

    first_name = list(paths_dict.keys())[0]
    first_path = stem + first_name
    first_columns = paths_dict[first_name]
    if index_type == 'str':
        # Using character based column indexes
        first_columns.append('Dates')
        base = pd.read_csv(first_path, usecols=first_columns)
    else:
        # Using numeric column indexes, dates = 0th column
        first_columns.append(0)
        base = pd.read_csv(first_path, usecols=first_columns)

    # Set the date column as standard date time format
    base['Date'] = pd.to_datetime(base['Date'], errors='coerce')

    # Remove first path name
    paths_dict.pop(first_name, None)
    paths = [stem + name for name in paths_dict.keys()]

    for path, key in zip(paths, paths_dict.keys()):
        # Find columns to read for each file
        columns = paths_dict[key]
        if index_type == 'str':
            # Using character based column indexes
            columns.append('Dates')
            data_in = pd.read_csv(path, usecols=columns)
        else:
            # Using numeric column indexes, dates = 0th column
            columns.append(0)
            data_in = pd.read_csv(path, usecols=columns)

        # Ensure all dates in a standard form
        data_in['Date'] = pd.to_datetime(data_in['Date'], errors='coerce')

        # Join new data to data frame using a left join
        base = base.merge(data_in, how='left', on='Date', suffixes=['_orig', '_'+key])

    return base


if __name__ == "__main__":
    stem = "Data Sets\\FTSEICA\\"

    names = {"BARC Historical Data.csv": [1],
             "BP Historical Data.csv": [1],
             "FTSE 100 Futures Historical Data.csv": [1],
             "FTSE 100 Historical Data.csv": [1, 2, 3, 4, 5],
             "LLOY Historical Data.csv": [1],
             "TW Historical Data.csv": [1],
             "Glen Historical Data.csv": [1],
             "United Kingdom 3-Month Bond Yield Historical Data.csv": [1],
             "United Kingdom 1-Month Bond Yield Historical Data.csv": [1],
             "VOD Historical Data.csv": [1],
             # "BCS.csv": [5],
             }

    test = load_data(stem, names, index_type='0')