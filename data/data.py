"""
Module for generating data points of the Lorenz-63 and Roessler systems. 
"""

import numpy as np
from scipy.integrate import odeint

def generate_lorenz_63(params=[10, 28, -8/3], initial_state=[1, 1, 1],
                        del_t=0.02, steps=1000, transient=500):
    '''
    Input:      params:         parameters of the Lorenz-63 system
                initial_state:  initial spatial conditions at time t=0
                del_t:          size of time step
                steps:          number of time steps
                transient:      only return values with index greater than or 
                                equal to washout
    
    Output:     a stepsx3 matrix represented as a numpy.ndarray
    '''
    # get (sigma, rho, beta) from params vector
    sigma, rho, beta = params
    
    # helper function for computing derivatives at each time iteration
    def f(state, t):
        '''
        Input: state vector in R3
        Output: derivative vector in R3 for L63 with passed parameters
        '''

        # get (x, y, z) components of the state vector
        x, y, z = state

        # return the value of the derivatives
        return (
            sigma * (y - x),
            x * (rho - z) - y,
            x * y + beta * z
        )
    
    # define length of simulation (time), i.e. number of time increments
    T = np.linspace(0, int(del_t * steps), steps)

    # integrate the system and return generated data
    all_data = odeint(func = f, y0 = initial_state, t = T)

    # washout time of 500 time steps
    return all_data[transient:]

# roessler
def generate_roessler(params=[0.1, 0.1, 18], initial_state=[1, 1, 1],
                        del_t=0.018, steps=1000, washout=500):
    '''
    Input:      params:         parameters of the Lorenz-63 system
                initial_state:  initial spatial conditions (time t=0)
                del_t:          size of time step
                steps:          number of time steps
                washout:        only return values with index greater than or 
                                equal to washout
    
    Output:     a stepsx3 matrix represented as a numpy.ndarray
    '''

    # get (a, b, c) from params vector
    a, b, c = params

    # helper function for computing derivatives at each time iteration
    def f(state, t):
        '''
        Input: state vector in R3
        Output: derivative vector in R3 for L63 with passed parameters
        '''

        # get (x, y, z) components of the state vector
        x, y, z = state

        # return the value of the derivatives
        return (
            - y - z,
            x + a * y,
            b + z * (x - c)
        )
    
    # define length of simulation (time)
    T = np.linspace(0, int(del_t * steps), steps)

    # integrate the system and return generated data
    all_data = odeint(func = f, y0 = initial_state, t = T)

    return all_data[washout:]

def lyapunov_to_steps(k: int, LAMBDA: np.double, delta_t: np.double):
    """
    A function to convert a discrete time scale into multiples of Lyapunov 
    exponents. 

    Args:
        k (int): the number of Lyapunov exponents desired.
        LAMBDA (np.double): the Lyapunov exponent.
        delta_t (np.double): the time step size.

    Returns:
        (int): the number of time steps equivalent to k Lyapunov time.
    """
    return int((k / LAMBDA) / delta_t)

if __name__ == '__main__':
    pass