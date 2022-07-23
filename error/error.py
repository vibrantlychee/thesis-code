import numpy as np

def RMSE(u, u_hat):
    '''
    RMSE between actual and predicted, normalised by variance of actual. 

    INPUTS:     u:      (np.array) true values of shape TxD
                u_hat:  (np.array) predicted values of shape TxD 
    '''
    T = u.shape[0]

    return np.sqrt(np.sum((u_hat - u) ** 2, axis=0) / T) / np.std(u)

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
    norms = np.array([np.linalg.norm(diff[t]) ** 2 for t in range(diff.shape[0])])
    norms_std = np.std(norms)

    return np.sqrt(delta_t * LAMBDA * sum(norms)) / norms_std

def griffith_epsilon(u: np.ndarray, u_hat: np.ndarray, delta_t: np.double, 
              LAMBDA: np.double) -> np.double:
    """
    """
    
    n = u.shape[0]
    t_i_range = np.arange(0, n, n // 50)

    errors = []
    for t_i in t_i_range:
        errors.append(griffith_epsilon_1(
            u=u,
            u_hat=u_hat,
            delta_t=delta_t,
            LAMBDA=LAMBDA,
            t_1=t_i
        ) ** 2)

    return np.sqrt(np.sum(errors) / 50)