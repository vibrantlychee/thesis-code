"""
Usage:
    python nodes.py [NUM_TRIALS] [LEN_N_RANGE]
For example:
    python nodes.py 100 6
represents simulating the number of reservoir nodes at 100, 200, 400, 800, 1600, 
3200. For each simulation, we perform 100 trials to smooth out the randomness.
"""

import sys
import os
from datetime import datetime

import numpy as np

import itertools
import multiprocessing

from reservoir import reservoir as res
from data import data
from error import error

################################################################################
###############################-----CONSTANTS-----##############################
################################################################################
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

NUM_TRIALS = int(sys.argv[1])
trial_range = range(NUM_TRIALS)
NUM_N_VALS = int(sys.argv[2])
N_range = [100 * (2 ** n) for n in range(NUM_N_VALS)]

# today's date
today = datetime.today().strftime("%Y%m%d")

training_data_ls = []
test_data_ls = []

################################################################################
############################-----HELPER FUNCTIONS-----##########################
################################################################################
# make a directory
def make_dir(dir):
    try:
        os.makedirs(dir)
        print("Created directory: {}".format(dir))
        return True
    except FileExistsError:
        return False

def generate_initial_states(num_trials, simulation_parameters):
    d = simulation_parameters["d"]
    initial_state_range = simulation_parameters["INITIAL_STATE_RANGE"]
    
    training_initial_states = STATE.uniform(-initial_state_range, initial_state_range, (num_trials, d))
    test_initial_states = STATE.uniform(-initial_state_range, initial_state_range, (num_trials, d))

    return training_initial_states, test_initial_states

training_initial_states, test_initial_states = generate_initial_states(
    num_trials=NUM_TRIALS,
    simulation_parameters=simulation_parameters
)

def generate_datasets(trial_id):
    training_data = data.generate_lorenz_63(
        initial_state=training_initial_states[trial_id],
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"],
        transient=simulation_parameters["WASHOUT"]
    )

    test_data = data.generate_lorenz_63(
        initial_state=test_initial_states[trial_id],
        del_t=simulation_parameters["DEL_T"],
        steps=simulation_parameters["STEPS"],
        transient=simulation_parameters["WASHOUT"]
    )
    print("\t\tDataset #{} generated.".format(trial_id))

    return training_data, test_data

def trial(trial_id, res_dim, training_data_ls, test_data_ls):
    trial_dir = "results/node_lengths/{}-{}-{}-{}/{}/{}/".format(today, NUM_TRIALS, N_range[0], N_range[-1], res_dim, trial_id)
    make_dir(trial_dir)
    
    hyperparams["N"] = res_dim

    # access datasets
    training_data = training_data_ls[trial_id]
    test_data = test_data_ls[trial_id]

    # write data to storage
    print("\t\tWriting datasets...", end="")
    np.savez_compressed(
        trial_dir + "data",
        training_data=training_data,
        test_data=test_data
    )
    print("complete.")

    d = training_data.shape[1]

    # generate reservoir inputs
    W_r = res.generate_W_r(hyperparams, (res_dim, res_dim), STATE)
    # print("\t\tWriting W_r...", end="")
    # np.savetxt(trial_dir + "W_r.out", W_r, delimiter=",")
    # print("complete.")

    W_in = res.generate_W_in(hyperparams, (res_dim, d), STATE)
    # print("\t\tWriting W_in...", end="")
    # np.savetxt(trial_dir + "W_in.out", W_in, delimiter=",")
    # print("complete.")
    
    # compute training reservoir
    training_res = res.generate_training_reservoir(
        data=training_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )
    # print("\t\tWriting training reservoir...", end="")
    # np.savetxt(trial_dir + "training_res.out", training_res, delimiter=",")
    # print("complete.")

    # compute output transformation
    W_out = res.generate_W_out(
        data=training_data,
        res=training_res,
        alpha=simulation_parameters["ALPHA"]
    )
    # print("\t\tWriting output transformation...", end="")
    # np.savetxt(trial_dir + "W_out.out", W_out, delimiter=",")
    # print("complete.")

    # compute forecast reservoir
    unmodified_forecast_res, forecast_res = res.generate_forecast_reservoir(
        r_0=np.dot(W_in, test_data[0]),
        data=test_data,
        hyperparams=hyperparams,
        W_r=W_r,
        W_in=W_in,
        W_out=W_out,
        delta_t=simulation_parameters["DEL_T"],
        adjust_for_symmetry=True
    )
    # print("\t\tWriting forecast reservoir...", end="")
    # np.savetxt(trial_dir + "forecast_res.out", forecast_res, delimiter=",")
    # print("complete.")

    # make prediction
    preds = res.readout_network(
        res=forecast_res,
        W_out=W_out
    )
    # print("\t\tWriting predictions...", end="")
    # np.savetxt(trial_dir + "preds.out", preds, delimiter=",")
    # print("complete.")

    # compute loss functions
    losses = np.array([
        # L2 error
        np.linalg.norm(error.RMSE(test_data, preds)),
        # Error over one Lyapunov time (Griffith et al)
        error.griffith_epsilon_1(test_data, preds, simulation_parameters["DEL_T"], LAMBDA)
    ])
    # print("\t\tWriting losses...", end="")
    # with open(trial_dir + "losses.out", 'w') as f:
    #     f.write(str(losses)[1:-1])
    # print("complete.")

    print("\t\tWriting results for reservoir Dimension: {}, Trial: #{} / {}, Loss: {}...".format(res_dim, trial_id, NUM_TRIALS, losses[1]), end="")
    np.savez_compressed(
        trial_dir + "results",
        W_r=W_r,
        W_in=W_in,
        W_out=W_out,
        preds=preds,
        losses=losses
    )
    print("complete.")

################################################################################
##############################-----SIMULATIONS-----#############################
################################################################################
if __name__ == '__main__':
    os.system('clear')
    
    # make directory for this test run
    dir = "results/node_lengths/{}-{}-{}-{}/".format(today, NUM_TRIALS, N_range[0], N_range[-1])
    make_dir(dir)
    
    print("Welcome.\nToday's date is: {}.\nIt is a good day for a simulation.\n".format(today))
    print("Below are the simulation parameters.")
    print("Reservoir dimensions tested: {}".format(str(N_range)[1:-1]))
    print("Number of trials per reservoir node value: {}".format(NUM_TRIALS))

    start = datetime.now()

    print("Beginning simulation...")
    
    # generate datasets for trials
    print("\tGenerating some datasets...")
    pool = multiprocessing.Pool()
    result = pool.map(generate_datasets, trial_range)
    pool.close()
    pool.join()
    print("\tcomplete.")

    training_data_ls = [result[i][0] for i in range(len(result))]
    test_data_ls = [result[i][1] for i in range(len(result))]

    # perform trials for each reservoir dimension
    print("\tPerforming trials...")
    for N in N_range:
        paramgrid = list(itertools.product(trial_range, [N], [training_data_ls], [test_data_ls]))
        # for x in paramgrid:
        #     print(x)
        # print(len(paramgrid))
        pool_2 = multiprocessing.Pool()
        pool_2.starmap(trial, paramgrid)
        pool_2.close()
        pool_2.join()

    # for N in N_range:
    #     for i in trial_range:
    #         trial(i, N, training_data_ls, test_data_ls)
    print("\tcomplete.")
    
    print("complete.\n")
    
    end = datetime.now()
    computation_time = (end - start).total_seconds() / 60

    print("Simulation time taken: {} mins".format(computation_time))
    sim_dir = "results/node_lengths/{}-{}-{}-{}/".format(today, NUM_TRIALS, N_range[0], N_range[-1])
    with open(sim_dir + "metadata.txt", 'w') as f:
        f.write("mins: " + str(computation_time) + "\n")