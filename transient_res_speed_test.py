import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "white"

import multiprocessing

from data import data
from reservoir import reservoir as res
from error import error

LAMBDA = 0.9056
NUM_TRIALS = int(sys.argv[1])

# reservoir dimension
N = 100

hyperparams = {
    'GAMMA': 7.7,
    'SIGMA': 0.81,
    'RHO_IN': 0.37,
    'K': 3,
    'RHO_R': 0.41
}

simulation_parameters = {
    "DEL_T": 0.02,          # time step size
    "STEPS": 30000,         # total steps
    "TRANSIENT": 10000,       # TRANSIENT 
    # n = STEPS - TRANSIENT
    "ALPHA": 0.001,         # Tikhonov regularisation constant
    "d": 3,
    "INITIAL_STATE_RANGE": 5
}

dir = "results/transient_res_speed_test/"

# with transient reservoir
def trial_with_transient_res(seed):
    start = datetime.now()
    
    state = np.random.RandomState(seed)
    
    training_ic = state.choice(np.linspace(-5, 5), 3)
    reservoir_transient_ic = state.choice(np.linspace(-5, 5), 3)

    training_data = data.generate_lorenz_63(
        initial_state=training_ic,
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"],
        transient=simulation_parameters["TRANSIENT"]
    )

    reservoir_transient_data = data.generate_lorenz_63(
        initial_state=reservoir_transient_ic,
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"],
        transient=simulation_parameters["TRANSIENT"]
    )

    W_in = res.generate_W_in(hyperparams=hyperparams, shape=(N, simulation_parameters["d"]), state=state)
    W_r = res.generate_W_r(hyperparams=hyperparams, shape=(N, N), state=state)

    training_res = res.generate_training_reservoir(
        data=training_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )

    W_out = res.generate_W_out(
        data=training_data,
        res=training_res,
        alpha=simulation_parameters["ALPHA"]
    )

    unmodified_transient_forecast_res, transient_forecast_res = res.generate_forecast_reservoir(
        r_0=np.dot(W_in, reservoir_transient_data[0]),
        data=reservoir_transient_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        W_out=W_out,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )

    transient_preds = res.readout_network(res=transient_forecast_res, W_out=W_out)

    test_ic = transient_preds[-1]
    test_data = data.generate_lorenz_63(
        initial_state=test_ic,
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"],
        transient=0
    )

    unmodified_forecast_res, forecast_res = res.generate_forecast_reservoir(
        r_0=unmodified_transient_forecast_res[-1],
        data=test_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        W_out=W_out,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )

    preds = res.readout_network(res=forecast_res, W_out=W_out)

    losses = {
        "RMSE": error.RMSE(test_data, preds),
        "griffith_epsilon_1": error.griffith_epsilon_1(test_data, preds, simulation_parameters["DEL_T"], LAMBDA),
        "griffith_epsilon": error.griffith_epsilon(test_data, preds, simulation_parameters["DEL_T"], LAMBDA),
    }

    end = datetime.now()

    time_taken = (end - start).total_seconds() / 60

    print("Writing results for with_transient_res: {} / {}...".format(seed, NUM_TRIALS), end="")
    np.savez_compressed(
        dir + "with_transient_res_" + str(seed),
        W_in=W_in,
        W_r=W_r,
        W_out=W_out,
        training_ic=training_ic,
        reservoir_transient_ic=reservoir_transient_ic,
        preds=preds,
        loss=losses,
        mins=time_taken
    )
    print("complete.")

# without transient reservoir
def trial_no_transient_res(seed):
    start = datetime.now()
    
    state = np.random.RandomState(seed)

    training_ic = state.choice(np.linspace(-5, 5), 3)

    lorenz_data = data.generate_lorenz_63(
        initial_state=training_ic,
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"] * 2,
        transient=simulation_parameters["TRANSIENT"]
    )
    
    n = lorenz_data.shape[0]

    training_data = lorenz_data[0 : n//2]
    test_data = lorenz_data[n//2 : ]

    W_in = res.generate_W_in(hyperparams=hyperparams, shape=(N, simulation_parameters["d"]), state=state)
    W_r = res.generate_W_r(hyperparams=hyperparams, shape=(N, N), state=state)

    unmodified_training_res = res.generate_training_reservoir(
        data=training_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=False
    )
    
    training_res = res.generate_training_reservoir(
        data=training_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )

    W_out = res.generate_W_out(
        data=training_data,
        res=training_res,
        alpha=simulation_parameters["ALPHA"]
    )

    unmodified_forecast_res, forecast_res = res.generate_forecast_reservoir(
        r_0=unmodified_training_res[-1],
        data=test_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        W_out=W_out,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )

    preds = res.readout_network(res=forecast_res, W_out=W_out)

    losses = {
        "RMSE": error.RMSE(test_data, preds),
        "griffith_epsilon_1": error.griffith_epsilon_1(test_data, preds, simulation_parameters["DEL_T"], LAMBDA),
        "griffith_epsilon": error.griffith_epsilon(test_data, preds, simulation_parameters["DEL_T"], LAMBDA),
    }

    end = datetime.now()

    time_taken = (end - start).total_seconds() / 60

    print("Writing results for no_transient_res: {} / {}...".format(seed, NUM_TRIALS), end="")
    np.savez_compressed(
        dir + "no_transient_res_" + str(seed),
        W_in=W_in,
        W_r=W_r,
        W_out=W_out,
        training_ic=training_ic,
        test_ic=test_data[0],
        preds=preds,
        loss=losses,
        mins=time_taken
    )
    print("complete.")

if __name__ == '__main__':
    cpus = int(multiprocessing.cpu_count() * (2/3))

    print("Beginning simulation.")
    print("Number of CPU cores for this script: {}".format(cpus))
    
    try:
        os.makedirs(dir)
    except:
        print("Directory already exists.")
    
    print("Starting trials with no transient reservoir.")
    pool = multiprocessing.Pool(cpus)
    pool.map(
        func=trial_no_transient_res,
        iterable=range(NUM_TRIALS)
    )
    pool.close()
    pool.join()
    print("Completed trials with no transient reservoir.")

    print("Starting trials with transient reservoir.")
    pool = multiprocessing.Pool(cpus)
    pool.map(
        func=trial_with_transient_res,
        iterable=range(NUM_TRIALS)
    )
    pool.close()
    pool.join()
    print("Completed trials with transient reservoir.")
