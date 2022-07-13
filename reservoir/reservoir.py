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

def generate_forecast_res(u, hyperparams, A, W_in, W_out, alpha=ALPHA):
    n = u.shape[0]
    R = hyperparams["R"]


    res = np.zeros((n, R))
    res[0] = np.dot(W_in, u[0])

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
        r[t] = next_training_res(
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