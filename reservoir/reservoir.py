# module for generating reservoir, training parameters, and predicting

import numpy as np

def generate_inputs(rho, s_in, R, D, seed):
    '''
    Given parameters rho, s_in, and a seed, returns the rotation and weight 
    matrices whose elements are randomly drawn from Uniform(0, 1). The 
    output matrices are rescaled so that their largest eigenvalues are less 
    than or equal to rho and s_in respectively. 
    
    INPUTS:     rho:    (float) spectral radius of rotation matrix A
                s_in:   (float) spectral radius of weight matrix W_in
                R:      (int) dimension of reservoir
                D:      (int) dimension of input signal
                seed:   (int) seed for numpy RNG

    OUTPUTS:    A:      (np.ndarray) matrix of shape (R, R)
                W_in:   (np.ndarray) matrix of shape (R, D+1)
    '''

    seed = np.random.RandomState(seed)

    A = seed.rand(R, R)
    W_in = seed.rand(R, D)

    # compute largest eigenvalue or singular value
    max_eigval_A = max(np.linalg.eigvals(A))
    max_eigval_W_in = max(np.linalg.svd(W_in)[1])

    # rescale A and W_in
    A = (rho / max_eigval_A) * A
    W_in = (s_in / max_eigval_W_in) * W_in

    # TO DO: sparseness

    return A, W_in

def next_res(r_prev, u_in, A, W_in):
    '''
    Given the previous reservoir state, rotation and weight matrices, and 
    hyperparameters, generates the next reservoir state. 
    
    INPUTS:     r_prev:     (np.array) previous reservoir state of shape (R, 1)
                u_in:       (np.array) input signal state of shape (D, 1)
                A:          (np.ndarray) rotation matrix of shape (R, R)
                W_in:       (np.ndarray) weight matrix of shape (R, D)
        
    OUTPUTS:    (np.array) next reservoir state of shape (R, 1)
    '''
    term = np.tanh(
        (np.array(np.matmul(np.asmatrix(A), np.asmatrix(r_prev).transpose())).flatten())
        + (np.array(np.matmul(np.asmatrix(W_in), np.asmatrix(u_in).transpose())).flatten())
    )

    return term

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
    D = u.shape[1]

    # generate rotation and weight matrices
    A, W_in = generate_inputs(rho=rho, s_in=s_in, R=R, D=D, seed=seed)
    # initialise reservoir with first signal
    r_0 = np.matmul(W_in, u[0])

    # allocate memory for reservoir
    r = np.ndarray((T, R))
    # set first reservoir state
    r[0] = r_0
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
    D = u.shape[1]
    
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


def predict(u_0, p, T, A, W_in, r_0=None):
    D = u_0.shape[0]

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

    prev_res = r_0
    for t in range(1, T, 1):
        curr_res = next_res(prev_res, u_hat[t-1], A, W_in)
        u_hat[t] = W_out(curr_res, p)
        prev_res = curr_res

    return u_hat
    

if __name__ == '__main__':
    pass