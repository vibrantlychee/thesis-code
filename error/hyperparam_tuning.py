from typing import Type, Optional, Union, Tuple

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from skopt import gp_minimize

import matplotlib.pyplot as plt 
plt.rcParams['figure.facecolor'] = "white"

from reservoir import reservoir as res
from data import data
from error import error

# define constants 
LAMBDA = 0.9056     # lorenz lyapunov exponent
SEED = 42
STATE = np.random.RandomState(SEED)

hyperparams = {
    'GAMMA': 7.7,
    'SIGMA': 0.81,
    'RHO_IN': 0.37,
    'K': 3,
    'RHO_R': 0.41,
    'N': 100
}

simulation_parameters = {
    "DEL_T": 0.02,          # time step size
    "STEPS": 30000,         # total steps
    "WASHOUT": 10000,       # washout 
    # n = STEPS - WASHOUT
    "ALPHA": 0.001,         # Tikhonov regularisation constant
    "d": 3,
    "INITIAL_STATE_RANGE": 5
}

