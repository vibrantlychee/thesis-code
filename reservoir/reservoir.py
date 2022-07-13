"""
Functions for generating a reservoir, training parameters, and predicting.
"""

import numpy as np

ALPHA = 1

def generate_A(rho: np.double, state: np.random.RandomState, shape: tuple, sparseness_coeff: np.double = 0.05) -> np.ndarray:
    """
    Generates the adjacency matrix A as specified. 

    Args:
        rho (np.double): the spectral radius of A.
        state (np.random.RandomState): a known RandomState for reproduceability. 
        shape (tuple): the shape of the adjacency matrix, R x R. 
        sparseness_coeff (np.double): controls the sparseness of A. 
            Must be a decimal between 0 and 1. Defaults to 0.05. A smaller 
            number means that A will be more sparse. 

    Returns:
        (np.ndarray): the generated adjacency matrix. 
    """
    # initialise as zero
    A = np.zeros(shape)

    # compute k, sparseness of the network
    k = int(sparseness_coeff * A.shape[0])
    for j in range(shape[1]):
        # choose the indices to be nonzero
        indices = state.choice(range(shape[0]), k, replace=False)
        for i in indices:
            # set nonzero elements to be 1
            A[i][j] = 1

    # scale the nonzero elements by Uniform(0, 1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if A[i][j] != 0:
                A[i][j] = A[i][j] * state.uniform(0, 1)

    # scale A such that its spectral radius is less than or equal to rho
    max_eigval = max(np.abs(np.linalg.eigvals(A)))
    A = (rho / max_eigval) * A

    return A

def generate_W_in(sigma: np.double, state: np.random.RandomState, shape: tuple) -> np.ndarray:
    """
    Shape R x D.
    """
    
    # initialise W_in
    W_in = np.zeros(shape)

    # draw a random permutation of integers 0,...,R-1
    perms = state.permutation(shape[0])

    j = 0
    for index in range(shape[0]):
        i = perms[index]
        W_in[i][j] = 1
        if (index + 1) % (shape[0] // shape[1]) == 0:
            j += 1

    # scale nonzero elements by a weight chosen randomly from Uniform(-sigma, sigma)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if W_in[i][j] != 0:
                W_in[i][j] = W_in[i][j] * state.uniform(-sigma, sigma)
    
    return W_in

def generate_inputs(rho, sigma, R, D, state, sparseness_coeff):
    '''
    Given parameters rho, s_in, and a seed, returns the rotation and weight 
    matrices whose elements are randomly drawn from Uniform(0, 1). The 
    output matrices are rescaled so that their largest eigenvalues are less 
    than or equal to rho and s_in respectively. 
    
    INPUTS:     rho:    (float) spectral radius of A
                sigma:  (float) weight of the nonzero elements of W_in
                R:      (int) dimension of reservoir
                D:      (int) dimension of input signal
                seed:   (int) seed for numpy RNG

    OUTPUTS:    A:      (np.ndarray) matrix of shape (R, R)
                W_in:   (np.ndarray) matrix of shape (R, D)
    '''
    A = generate_A(rho, state, (R, R), sparseness_coeff)
    W_in = generate_W_in(sigma, state, (R, D))

    return A, W_in

def next_training_res(r_prev, u_in, A, W_in, alpha):
    '''
    Given the previous reservoir state, rotation and weight matrices, and 
    hyperparameters, generates the next reservoir state. 
    
    INPUTS:     r_prev:     (np.array) previous reservoir state of shape (R, 1)
                u_in:       (np.array) input signal state of shape (D, 1)
                A:          (np.ndarray) rotation matrix of shape (R, R)
                W_in:       (np.ndarray) weight matrix of shape (R, D)
        
    OUTPUTS:    (np.array) next reservoir state of shape (R, 1)
    '''
    return (1 - ALPHA) * r_prev + ALPHA * np.tanh(
        np.dot(A, r_prev) + np.dot(W_in, u_in)
    )

def generate_training_reservoir(u: np.ndarray, hyperparams: dict, state: np.random.RandomState, 
                                sparseness_coeff=0.05, alpha=ALPHA):
    """
    Shape n x R. 
    """
    # dimensions
    n = u.shape[0]        # number of reservoir states
    D = u.shape[1]      # dimension of input signal

    # parse hyperparams
    rho = hyperparams["RHO"]
    sigma = hyperparams["SIGMA"]
    R = hyperparams["R"]        # number of reservoir nodes
    
    # generate internals
    A, W_in = generate_inputs(rho, sigma, R, D, state, sparseness_coeff)

    # initialise reservoir
    res = np.zeros((n, R))

    res[0] = np.dot(W_in, u[0])
    for i in range(n):
        if i == 0:
            continue

        res[i] = next_training_res(
            r_prev=res[i-1],
            u_in=u[i],
            A=A,
            W_in=W_in,
            alpha=alpha
        )

    return res, A, W_in

def modify_node(node):
    N = node.shape[0]
    
    for i in range(N):
        if i % 2 == 0:
            node[i] = node[i] ** 2

    return node

def generate_W_out(u: np.ndarray, res: np.ndarray, hyperparams: dict):
    """
    Output shape is D x R.
    """
    # dimensions
    R = res.shape[1]
    # tikhonov
    epsilon = hyperparams["EPSILON"]

    # modification by squaring even indexed elements of each reservoir node
    for i in range(res.shape[0]):
        res[i] = modify_node(res[i])

    return np.dot(np.dot(np.linalg.inv(np.dot(res.transpose(), res) + epsilon * np.identity(R)), res.transpose()), u).transpose()

def generate_forecast_res(r_0, u, hyperparams, A, W_in, W_out, alpha=ALPHA):
    n = u.shape[0]
    R = hyperparams["R"]


    res = np.zeros((n, R))
    # res[0] = np.dot(W_in, u[0])
    res[0] = r_0

    for i in range(n):
        if i == 0:
            continue
        
        res[i] = next_training_res(
            r_prev=res[i-1],
            u_in=np.dot(W_out, res[i-1]),
            A=A,
            W_in=W_in,
            alpha=alpha
        )
    
    return res

def output_node(W_out, node):
    return np.dot(W_out, node)

def output_series(W_out, res):
    n = res.shape[0]
    D = W_out.shape[0]
    preds = np.zeros((n, D))
    for i in range(n):
        preds[i] = output_node(W_out, res[i])

    return preds

if __name__ == '__main__':
    pass