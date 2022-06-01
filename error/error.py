import numpy as np

def RMSE(u, u_hat):
    '''
    INPUTS:     u:      (np.array) true values of shape TxD
                u_hat:  (np.array) predicted values of shape TxD 
    '''
    T = u.shape[0]

    return np.sqrt(np.sum((u_hat - u) ** 2, axis=0) / T)