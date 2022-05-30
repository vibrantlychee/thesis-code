################################################################################
#####-MODULE-FOR-DATA-GENERATION-###############################################
################################################################################

import numpy as np
from scipy.integrate import odeint

# lorenz-63
def generate_lorenz_63(params=[10, 28, 8/3], initial_state=[1, 1, 1],
                        del_t=0.02, t_max=1000):
    '''
    Input:      params:         parameters of the Lorenz-63 system
                initial_state:  initial spatial conditions at time t=0
                del_t:          size of time step
                t_max:          maximum time value, which together with del_t 
                                determines the number of time increments (T)
    
    Output:     a (t_max / del_t)x3 matrix represented as a numpy.ndarray
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
            x * y - beta * z
        )
    
    # define length of simulation (time), i.e. number of time increments
    T = np.arange(0, t_max, del_t)

    # integrate the system and return generated data
    all_data = odeint(func = f, y0 = initial_state, t = T)

    return all_data

# roessler
def generate_roessler(params=[0.1, 0.1, 18], initial_state=[1, 1, 1],
                        del_t=0.3, t_max=1000):
    '''
    Input:      params:         parameters of the Lorenz-63 system
                initial_state:  initial spatial conditions (time t=0)
                del_t:          size of time step
                t_max:          maximum time_value, which together with del_t 
                                determines the number of time increments (T)
    
    Output:     a (t_max / del_t)x3 matrix represented as a numpy.ndarray
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
    T = np.arange(0, t_max, del_t)

    # integrate the system and return generated data
    all_data = odeint(func = f, y0 = initial_state, t = T)

    if t_min == 0:
        return all_data
    else:
        return all_data[int(t_min * len(T)):]

if __name__ == '__main__':
    pass