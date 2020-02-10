import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def normalise(data):
    # Normalises a data series by subtracting mean and dividing by standard deviation
    mean = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)
    return (data - mean) / stds


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
    return np.log(time_series[1:]) - np.log(time_series[:-1])


def load_data(stem, paths_dict, index_type='str'):
    # Loads all csvs based on stem and paths dictionary
    # Assumes first path has correct data length (by dates)
    # Returns m x N data frame with datetime as the first row

    first_name = list(paths_dict.keys())[0]
    first_path = stem + first_name
    first_columns = paths_dict[first_name]
    if index_type == 'str':
        # Using character based column indexes
        first_columns.append('Date')
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
            columns.append('Date')
            data_in = pd.read_csv(path, usecols=columns)
        else:
            # Using numeric column indexes, dates = 0th column
            columns.append(0)
            data_in = pd.read_csv(path, usecols=columns)

        # Ensure all dates in a standard form
        data_in['Date'] = pd.to_datetime(data_in['Date'], errors='coerce')

        # Join new data to data frame using an inner join to avoid NaN
        base = base.merge(data_in, how='inner', on='Date', suffixes=['_orig', '_'+key])

    # Make time series increasing chronology
    base_sorted = base.sort_values(by='Date')

    # Transpose dataframe to make it m x N
    base_m_n = base_sorted.transpose()

    # Remove any NaN columns which have made it through
    return base_m_n.dropna(axis=1)


def is_normal(data):
    # Useful for quickly checking if a data series is normal or not
    kurtosis_list = None
    if type(data) == np.ndarray:
        # Data is a numpy array
        if data.ndim == 1:
            # Data is a 1d numpy array so treat like a list
            kurtosis_list = kurtosis(data)
            kurtosis_min_val = abs(kurtosis_list)
        else:
            short_dim_idx = np.argmin(data.shape)
            size = data.shape[short_dim_idx]

            if short_dim_idx == 0:
                # Rows is the short side
                kurtosis_list = np.array([kurtosis(data[j, :]) for j in range(size)])
                kurtosis_min_val = np.min(abs(kurtosis_list))
            elif short_dim_idx == 1:
                # Columns are the short side
                kurtosis_list = np.array([kurtosis(data[:, j]) for j in range(size)])
                kurtosis_min_val = np.min(abs(kurtosis_list))
            else:
                raise Exception("Unexpected data dimension provided")

    elif type(data) == list:
        # Data is a list
        kurtosis_list = kurtosis(data)
        kurtosis_min_val = abs(kurtosis_list)

    else:
        kurtosis_list = None
        raise TypeError("Data Type not expected, please provide list or numpy array")

    # Check the smallest absolute kurtosis is far from 0
    if kurtosis_min_val < 1:
        print("Warning, kurtosis is", kurtosis_min_val)

    return kurtosis_list


def comp_sign(r):
    # Computes the sign of r
    if isinstance(r, (list, np.ndarray)):
        # Assume an array
        neg_indexes = np.where(r < 0)
        pos_indexes = np.where(r > 0)
        r_out = np.zeros(len(r))
        r_out[neg_indexes] = -1
        r_out[pos_indexes] = 1
        return r_out
    else:
        # Assume it is a single value
        if r > 0:
            return 1
        if r == 0:
            return 0
        else:
            return -1


def scale_uni(lower, upper):
    # Scales a [0,1) uniform random variable to any range
    # Output is addition factor then division factor
    return lower/(upper-lower), 1/(upper-lower)


def moving_average(data, n=3):
    # Calculates the n step moving average
    # Modified for multi dimension
    # https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy/54628145
    ret = np.cumsum(data, dtype=float, axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


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