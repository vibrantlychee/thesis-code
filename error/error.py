import numpy as np

def RMSE(u, u_hat):
    '''
    INPUTS:     u:      (np.array) true values of shape TxD
                u_hat:  (np.array) predicted values of shape TxD 
    '''
    T = u.shape[0]

    return np.sqrt(np.sum((u_hat - u) ** 2, axis=0) / T)

def griffith_epsilon_1(u: np.ndarray, u_hat: np.ndarray, delta_t: np.double, 
              LAMBDA: np.double, t_1: np.double = 0) -> np.double:
    """
    Computes the root-mean-square-error (RMSE) as defined in Griffith et. al.
    (2019). Should only be used when input signal is a dynamical system 
    as LAMBDA represents the Lyapunov exponent of the system. 
    
    Args:
        u (np.ndarray): actual signal.
        u_hat (np.ndarray): predicted signal.
        delta_t (np.ndarray): Size of time step. 
        LAMBDA (np.ndarray): Lyapunov exponent of the input signal. 

    Returns:
        (np.double): 

    """
    end = int((t_1 + 1 / LAMBDA) / delta_t)
    diff = u[t_1: end] - u_hat[t_1: end]
    norms = np.array([np.linalg.norm(diff[t]) for t in range(diff.shape[0])])

    return np.sqrt(delta_t * LAMBDA * sum(norms))