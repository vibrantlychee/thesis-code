"""
Methods for reservoir construction, output training, forecasting, and 
hyperparameter tuning.
"""

import numpy as np

def generate_W_in(hyperparams: dict, shape: tuple, seed: int) -> np.ndarray:
    """
    Given hyperparameters, generates the matrix W_in, representing connections 
    between the input signal and nodes within the reservoir. 

    Args:
        hyperparams (dict): dictionary of hyperparameters.
            Must have the keys 'SIGMA' and 'RHO_IN'. 
        shape (tuple): tuple of two nonnegative integers, representing N x d.
        seed (int): seed for use when sampling using numpy.
            Consider keeping track of the seed used to ensure results are 
            reproducible. 

    Returns:
        (np.ndarray): matrix of shape N x d.
    """
    
    # parse hyperparameters
    SIGMA = hyperparams['SIGMA']
    RHO_IN = hyperparams['RHO_IN']

    # initialise output matrix of given shape
    W_in = np.ndarray(shape)

    # set random state using given seed
    state = np.random.RandomState(seed)

    # fill in connections in W_in 
    for i in range(W_in.shape[0]):
        for j in range (W_in.shape[1]):
            # whether or not connection exists is Bernoulli(SIGMA)
            W_in[i][j] = (state.uniform() < SIGMA) * 1
            # if connection exists, its weight is Normal(0, RHO_IN ** 2)
            if W_in[i][j] == 1:
                W_in[i][j] = (state.normal(0, RHO_IN ** 2))

    return W_in


def generate_W_r(hyperparams: dict, shape: tuple, seed: int):
    """
    Given hyperparameters, generates the matrix W_r, representing connections
    between reservoir nodes in the network. 
    
    Args:
        hyperparams (dict): dictionary of hyperparameters. 
            Must have the keys 'K' and 'RHO_R'.
        shape (tuple): tuple of two nonnegative integers, representing N x N.
        seed (int): seed for use when sampling using numpy.
            Consider keeping track of the seed used to ensure results are 
            reproducible. 

    Returns:
        (np.ndarray): matrix of shape N x N.
    """

    # parse hyperparameters
    K = hyperparams["K"]
    RHO_R = hyperparams["RHO_R"]

    # initialise output matrix of given shape
    W_r = np.ndarray(shape)

    # set random state using given seed
    state = np.random.RandomState(seed)

    # fill in connections in W_r
    for i in range(W_r.shape[0]):
        # choose k nodes without replacement
        choices = state.choice(W_r.shape[0], size=K, replace=False)
        
        for j in range(W_r.shape[1]):
            if j in choices:
                W_r[i][j] = state.normal(0, 1)
            else:
                W_r[i][j] = 0

    # spectral radius of W_r
    spectral_radius = max(abs(np.linalg.eigvals(W_r))) 

    # rescale W_r so that its spectral radius is equal to RHO_R
    W_r = (RHO_R / spectral_radius) * W_r

    return W_r

def next_res(r_prev: np.ndarray, u_prev: np.ndarray, hyperparams: dict, W_r: np.ndarray, W_in: np.ndarray, delta_t: np.double):
    """
    Computes the dynamics of the reservoir.

    Args:
        r_prev (np.ndarray): the previous reservoir state, of shape N x 1.
        u_prev (np.ndarray): the previous input signal, of shape d x 1.
        hyperparams (dict): dictionary of hyperparameters.
            Must have the key 'GAMMA'. A sensible range for GAMMA is 7 - 11. 
        W_r (np.ndarray): the adjacency matrix of the internal connection 
            network, of shape N x N. 
        W_in (np.ndarray): the adjacency matrix of the input signal to node 
            network, of shape N x d.  
        delta_t (np.double): the size of the time step. 
            Note that this should be the same as the time step used to integrate
            the dynamical system being forecasted (i.e. the input signal). 
    
    Returns:
        (np.ndarray): the next reservoir state, of shape N x 1. 
    """

    # parse hyperparameter
    GAMMA = hyperparams["GAMMA"]
    
    return r_prev + delta_t * ((-GAMMA) * r_prev) + GAMMA * np.tanh(np.dot(W_r, r_prev) + np.dot(W_in, u_prev))

def generate_reservoir(u, rho, s_in, R, seed):
    '''
    Given input signal data u and hyperparameters, generates the entire 
    reservoir over all time and outputs as a np.ndarray of shape (T, R). 

    INPUTS:     u:          (np.ndarray) input signal data of shape (T, D)
                alpha:      (float) leaking rate
                rho:        (float) spectral radius of rotation matrix A
                s_in:       (float) spectral radius of weight matrix W_in
                R:          (int)   number of nodes in reservoir network
                seed:       (int) seed for numpy RNG

    OUTPUTS:    (np.ndarray) the reservoir as a matrix of shape (T, R), where 
                for any t in [0, T], the t-th row represents the reservoir 
                state at time t.
    '''
    # extract dimensions
    T = u.shape[0]
    try:
        D = u.shape[1]
    except IndexError:
        D = 1

    # generate rotation and weight matrices
    A, W_in = generate_inputs(rho=rho, s_in=s_in, R=R, D=D, seed=seed)
    # initialise reservoir with first signal
    r_0 = None
    if D > 1:
        r_0 = np.matmul(W_in, u[0])
    elif D == 1:
        r_0 = u[0] * W_in 

    # allocate memory for reservoir
    r = np.ndarray((T, R))
    # set first reservoir state
    r[0] = list(r_0)
    # set iterated reservoir states
    for t in range(1, T):
        r[t] = next_res(
            r_prev=r[t-1],
            u_in=u[t-1],
            A=A,
            W_in=W_in
        )

    return r, A, W_in

def train_p(u, rho, s_in, R, beta, seed):
    '''

    '''
    # dimensions of data
    T = u.shape[0]
    try:
        D = u.shape[1]
    except:
        D = 1
    
    # generate reservoir
    r, A, W_in = generate_reservoir(u=u, rho=rho, s_in=s_in, R=R, seed=seed)
    
    # construct design matrix
    X = np.concatenate((r, r**2), axis=1)
    Id = np.identity(X.transpose().shape[0])

    # solve for W_out using ridge regression with regularisation parameter beta
    p = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X) + beta * Id), X.transpose()), u)
    # W_out = np.matmul(Y_target, np.matmul(X.transpose(), np.linalg.inv(
    #     np.matmul(X, X.transpose()) + beta * np.identity(X.shape[0])
    # )))

    P_1 = p[:R]
    P_2 = p[R:]

    p = (P_1, P_2)

    return p, r, A, W_in

# readout
def W_out(r_t, p):
    (P_1, P_2) = p

    return np.matmul(r_t, P_1) + np.matmul(r_t ** 2, P_2)


def predict(u_0, p, T, A, W_in, r_0=None, full=False):
    try:
        D = u_0.shape[0]
    except IndexError:
        D = 1

    # initialise output
    u_hat = np.ndarray((T, D))
    # we are given first state
    u_hat[0] = u_0

    # by default, initialise reservoir at r_0
    if D > 1:
        if r_0 == None:
            r_0 = np.matmul(W_in, u_0)
    elif D == 1:
        if r_0 == None:
            r_0 = (W_in * u_0).flatten()

    full_res = []
    prev_res = r_0
    for t in range(1, T, 1):
        full_res.append(prev_res)
        curr_res = next_res(prev_res, u_hat[t-1], A, W_in)
        u_hat[t] = W_out(curr_res, p)
        prev_res = curr_res

    if D == 1:
        return u_hat.flatten()
        
    if full:
        return u_hat, np.array(full_res)
    
    return u_hat
    

if __name__ == '__main__':
    pass