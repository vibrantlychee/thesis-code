"""
Methods for reservoir construction, output training, forecasting, and 
hyperparameter tuning.
"""

import numpy as np
from typing import Type, Optional, Union, Tuple

def generate_W_in(hyperparams: dict, shape: tuple, state: np.random.RandomState) -> np.ndarray:
    """
    Given hyperparameters, generates the matrix W_in, representing connections 
    between the input signal and nodes within the reservoir. 

    Args:
        hyperparams (dict): dictionary of hyperparameters.
            Must have the keys 'SIGMA' and 'RHO_IN'. 
            Good ranges for the hyperparameters are: 'SIGMA': 0.1 - 1.0, 
            'RHO_IN': 0.3 - 1.5 (Griffith et. al.).
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

    # fill in connections in W_in 
    for i in range(W_in.shape[0]):
        for j in range (W_in.shape[1]):
            # whether or not connection exists is Bernoulli(SIGMA)
            W_in[i][j] = (state.uniform() < SIGMA) * 1
            # if connection exists, its weight is Normal(0, RHO_IN ** 2)
            if W_in[i][j] == 1:
                W_in[i][j] = (state.normal(0, RHO_IN ** 2))

    return W_in


def generate_W_r(hyperparams: dict, shape: tuple, state: np.random.RandomState) -> np.ndarray:
    """
    Given hyperparameters, generates the matrix W_r, representing connections
    between reservoir nodes in the network. 
    
    Args:
        hyperparams (dict): dictionary of hyperparameters. 
            Must have the keys 'K' and 'RHO_R'.
            Good ranges for the hyperparameters are: 'K': 1 - 5,
            'RHO_R': 0.3 - 1.5 (Griffith et. al.).
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

def modify_node(node: np.ndarray, adjust_for_symmetry: bool = True):
    """
    Modifies a given reservoir node to better account for the symmetry of the 
    input signal (e.g. a Lorenz system). 
    
    Args:
        node (np.ndarray): reservoir state of shape N x 1.
        adjust_for_symmetry (bool): Whether or not a transformation is required 
            on the reservoir in order to account for any symmetry in the 
            underlying input signal. 

            Defaults to True. 

    Returns:
        (np.ndarray): modified reservoir state of shape N x 1.
            If adjust_for_symmetry is False, then the node is returned 
            unchanged.

            If adjust_for_symmetry is True, then components whose index is 
            greater than N // 2 have been squared. Components whose index is 
            less than or equal to N // 2 remain unchanged.
    """
    
    if not(adjust_for_symmetry):
        return node

    N = node.shape[0]
    modified_node = np.ndarray(N)
    for i in range(N):
        if i <= N / 2:
            modified_node[i] = node[i]
        else:
            modified_node[i] = node[i] ** 2
    
    return modified_node

def next_training_node(r_prev: np.ndarray, u_prev: np.ndarray, 
                       hyperparams: dict, W_r: np.ndarray, 
                       W_in: np.ndarray, delta_t: np.double) -> np.ndarray:
    """
    Computes the dynamics of the reservoir in the training stage. That is, 
    the next reservoir stage is dependent on both the previous reservoir state 
    and an input signal. 

    Args:
        r_prev (np.ndarray): the previous reservoir state, of shape N x 1.
        u_prev (np.ndarray): the previous input signal, of shape d x 1.
        hyperparams (dict): dictionary of hyperparameters.
            Must have the key 'GAMMA'. 
            A good range for GAMMA is 7 - 11 (Griffith et. al.). 
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
    
    return r_prev + delta_t * (((-GAMMA) * r_prev) + GAMMA * np.tanh(np.dot(W_r, r_prev) + np.dot(W_in, u_prev)))

def generate_training_reservoir(data: np.ndarray, hyperparams: dict, 
                       W_r: np.ndarray, W_in: np.ndarray, delta_t: np.double,
                       adjust_for_symmetry: bool = True) -> np.ndarray:
    """
    Computes the training reservoir network given the generating training data 
    points. 

    Args:
        data (np.ndarray): matrix of input signals of shape n x d.
        hyperparams (dict): dictionary of hyperparameters.
            Must have the keys 'GAMMA', 'SIGMA', 'RHO_IN', 'K', 'RHO_R'. 
            Good ranges for the hyperparameters are: GAMMA: 7 - 11, 
            'SIGMA': 0.1 - 1.0, 'RHO_IN': 0.3 - 1.5, 'K': 1 - 5,
            'RHO_R': 0.3 - 1.5 (Griffith et. al.).
        W_r (np.ndarray): the adjacency matrix of the internal connection 
            network, of shape N x N. 
        W_in (np.ndarray): the adjacency matrix of the input signal to node 
            network, of shape N x d. 
        delta_t (np.double): the size of the time step. 
            Note that this should be the same as the time step used to integrate
            the dynamical system being forecasted (i.e. used to generate the 
            input signal). 
        adjust_for_symmetry (bool): Whether or not a transformation is required 
            on the reservoir in order to account for any symmetry in the 
            underlying input signal. 

            Defaults to True. Refer to modify_node() for how this adjustment is 
            performed. 

    Returns:
        (np.ndarray): the generated reservoir network.
    """
    # compute important dimensions
    n = data.shape[0]      # time steps
    N = W_r.shape[0]       # reservoir dimension

    # initialise echo state network
    res = np.ndarray((n, N))
    # fill in echo state network
    for i in range(n):
        # first reservoir is unique (as it has no previous state)
        if i == 0:
            res[i] = np.dot(W_in, data[0])
        # next reservoir is dependent on reservoir dynamics
        else:
            res[i] = next_training_node(
                r_prev=res[i-1],
                u_prev=data[i-1],
                hyperparams=hyperparams,
                W_r=W_r,
                W_in=W_in,
                delta_t=delta_t
            )

    return np.array([modify_node(res[i], adjust_for_symmetry) for i in range(n)])


def generate_W_out(data: np.ndarray, res: np.ndarray, alpha: np.double):
    """
    Given the input signal and the reservoir, trains the linear transformation 
    of the output layer using Tikhonov (ridge) regularised regression. 

    Args:
        data (np.ndarray): the input signal, of shape n x d.
        res (np.ndarray): the reservoir network, of shape n x N.
        alpha (np.double): the Tikhonov regularisation constant. 

    Returns:
        (np.ndarray) the output linear transformation, of shape d x N.
    """
    N = res.shape[1]

    return np.dot(np.dot(np.linalg.inv(np.dot(res.transpose(), res) + alpha * np.identity(N)), res.transpose()), data).transpose()
    
def next_forecast_node(r_prev: np.ndarray, hyperparams: dict, W_r: np.ndarray,
                       W_in: np.ndarray, W_out: np.ndarray, delta_t: np.double, 
                       adjust_for_symmetry: bool = True) -> np.ndarray:
    """
    Computes the dynamics of the reservoir in the forecast stage. That is, 
    the next reservoir node is dependent only on the previous reservoir state. 
    In particular, the next reservoir node does not depend on any incoming 
    input signal. 

    Args:
        r_prev (np.ndarray): the previous reservoir state, of shape N x 1.
        hyperparams (dict): dictionary of hyperparameters.
            Must have the key 'GAMMA'. 
            A good range for GAMMA is 7 - 11 (Griffith et. al.). 
        W_r (np.ndarray): the adjacency matrix of the internal connection 
            network, of shape N x N. 
        W_in (np.ndarray): the adjacency matrix of the input signal to node 
            network, of shape N x d.  
        W_out (np.ndarray): the output linear transformation. 
        delta_t (np.double): the size of the time step. 
            Note that this should be the same as the time step used to integrate
            the dynamical system (input signal) being forecasted.
        adjust_for_symmetry (bool): Whether or not a transformation is required 
            on the reservoir in order to account for any symmetry in the 
            underlying input signal. 

            Defaults to True. Refer to modify_node() for how this adjustment is 
            performed. 
    
    Returns:
        (np.ndarray): the next reservoir state, of shape N x 1. 

    """
    GAMMA = hyperparams["GAMMA"]

    return r_prev + delta_t * (((-GAMMA) * r_prev) + GAMMA * np.tanh(np.dot(W_r, r_prev) + np.dot(W_in, np.dot(W_out, modify_node(r_prev, adjust_for_symmetry)))))

def generate_forecast_reservoir(r_0: np.ndarray, data: np.ndarray, 
                                hyperparams: dict, W_r: np.ndarray,
                                W_in: np.ndarray, W_out: np.ndarray, 
                                delta_t: np.double, 
                                adjust_for_symmetry: bool = True) -> np.ndarray:
    """
    Computes the forecast reservoir network given test data points. 

    Args: 
        data (np.ndarray): matrix of input signals of shape n x d.
        hyperparams (dict): dictionary of hyperparameters.
            Must have the keys 'GAMMA', 'SIGMA', 'RHO_IN', 'K', 'RHO_R'. 
            Good ranges for the hyperparameters are: GAMMA: 7 - 11, 
            'SIGMA': 0.1 - 1.0, 'RHO_IN': 0.3 - 1.5, 'K': 1 - 5,
            'RHO_R': 0.3 - 1.5 (Griffith et. al.).
        W_r (np.ndarray): the adjacency matrix of the internal connection 
            network, of shape N x N. 
        W_in (np.ndarray): the adjacency matrix of the input signal to node 
            network, of shape N x d. 
        W_out (np.ndarray): the output layer matrix, computed using Tikhonov 
            regularised regression in the training stage, of shape d x N. 
        delta_t (np.double): the size of the time step. 
            Note that this should be the same as the time step used to integrate
            the dynamical system being forecasted (i.e. used to generate the 
            input signal). 
        adjust_for_symmetry (bool): Whether or not a transformation is required 
            on the reservoir in order to account for any symmetry in the 
            underlying input signal. 

            Defaults to True. Refer to modify_node() for how this adjustment is 
            performed. 

    Returns:
        (np.ndarray): the generated reservoir network. 
    """
    # dimensions
    n = data.shape[0]
    N = W_r.shape[0]
    
    # initialise reservoir network
    res = np.ndarray((n, N))
    for i in range(n):
        if i == 0:
            res[i] = r_0
        else:
            res[i] = next_forecast_node(
                r_prev=res[i-1],
                hyperparams=hyperparams,
                W_r=W_r,
                W_in=W_in,
                W_out=W_out,
                delta_t=delta_t,
                adjust_for_symmetry=adjust_for_symmetry
            )

    return res, np.array([modify_node(res[i]) for i in range(n)]) 

def readout_node(node: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Given a reservoir node and output linear transformation, computes the 
    predicted signal.

    Args:
        node (np.ndarray): the reservoir node, of shape N x 1.
        W_out (np.ndarray): the output linear transformation, of shape d x N.

    Returns:
        (np.ndarray): the predicted signal, of shape d x 1. 
    """

    return np.dot(W_out, node)

def readout_network(res: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Given a reservoir network and output linear transformation, computes the 
    predicted signal series. 

    Args:
        res (np.ndarray): the reservoir network, of shape n x N.
        W_out (np.ndarray): the output linear transformation, of shape d x N.

    Returns:
        (np.ndarray): the predicted signal series, of shape n x d. 
    """
    return np.array([readout_node(res[i], W_out) for i in range(res.shape[0])])

##############
#### OLD  ####
##############
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
        curr_res = next_training_node(prev_res, u_hat[t-1], A, W_in)
        u_hat[t] = W_out(curr_res, p)
        prev_res = curr_res

    if D == 1:
        return u_hat.flatten()
        
    if full:
        return u_hat, np.array(full_res)
    
    return u_hat
    

if __name__ == '__main__':
    pass