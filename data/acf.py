import numpy as np

# a helper function to parse which axis to compute acfs
def parse_axis(axis):
        if axis == "x":
            return 0
        elif axis == "y":
            return 1
        else:
            return 2

# autocorrelation
def compute_acf(data, axis1, axis2, l):
    '''
    Inputs:     data:   Tx3 matrix generated from odeint, e.g. l63 or roessler
                axis1:  (str) spatial dimension of first time series (i.e. x, y, or z)
                axis2:  (str) spatial dimension of second time series (i.e. x, y, or z)
                l:      (float) time delay, 0 < l < data.shape[0]

    Output:     real number, representing autocorrelation between axis1 and 
                axis2 for the given time lag value (l)
    '''
    # obtain parameter values
    T = data.shape[0]       # number of time increments

    # get two spatial series
    series1 = data[:, parse_axis(axis1)]
    series2 = data[:, parse_axis(axis2)]
    
    # mean of two spatial dimensions
    mean_1 = np.mean(series1)
    mean_2 = np.mean(series2)
    
    # get the two spatial dimensions we are comparing correlations
    # subtract mean from series1
    series1 = (series1 - mean_1) ** 2
    series2 = (series2 - mean_2) ** 2

    # # set up zero matrix
    # lagged_series2 = np.zeros(series1.shape)
    # # subtract mean and apply time lag
    # for i in range(0, T-l, 1):
    #     lagged_series2[i] = ((series2 - mean_2)[i+l]) ** 2

    # compute autocorrelation value for given l
    return np.dot(series1[0:T-l], series2[l:T]) / T
    # return np.dot(series1, lagged_series2) / T

def compute_acfs(data, axis1, axis2):
    '''
    Inputs:     data:   Tx3 matrix generated from odeint, e.g. l63 or roessler
                axis1:  (str) spatial dimension of first time series (i.e. x, y, or z)
                axis2:  (str) spatial dimension of second time series (i.e. x, y, or z)

    Output:     array of floats, representing autocorrelation between axis1 and 
                axis2 for all possible time delays l, 0 <= l <= T
    '''

    # extract shape of data
    T, D = data.shape
    l_range = range(0, T, 1)
    
    out = [compute_acf(data, axis1, axis2, l) for l in l_range]

    return np.array(out)