import itertools
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = "white"

from reservoir import reservoir as res
from data import data
from error import error

LAMBDA = 0.9056     # lorenz lyapunov exponent
DEL_T = 0.02

TRIAL_RANGE = range(100)
N_RANGE = [100, 200, 400, 800, 1600]

def generate_dir(node_length, trial_num):
    dir = "results/node_lengths/20220718-100-100-3200/{}/{}/".format(node_length, trial_num)
    if node_length == 1600:
        dir = "results/node_lengths/20220719-100-100-3200/{}/{}/".format(node_length, trial_num)

    return dir

def load_data(node_length, trial_num):
    dir = generate_dir(node_length, trial_num)

    lorenz_data = np.load(dir + "data.npz")
    results = np.load(dir + "results.npz")

    return lorenz_data, results

all_trials = {}
for node_length in N_RANGE:
    trials = {}
    for trial_num in TRIAL_RANGE:        
        trials[trial_num] = load_data(node_length, trial_num)

    all_trials[node_length] = trials

def write_trial_data(res_dim, trial_num, start=0, end=15):
    dir = generate_dir(res_dim, trial_num)
    
    # access the data and results
    model_data = all_trials[res_dim][trial_num][0]
    results = all_trials[res_dim][trial_num][1]

    # parse the data and results
    training_data = model_data["training_data"]
    test_data = model_data["test_data"]
    W_r = results["W_r"]
    W_in = results["W_in"]
    W_out = results["W_out"]
    preds = results["preds"]
    losses = results["losses"]

    # set up plotting parameters
    N = res_dim
    start = data.lyapunov_to_steps(0, LAMBDA, DEL_T)
    end = data.lyapunov_to_steps(15, LAMBDA, DEL_T)
    x_range = LAMBDA * DEL_T * np.array(range(start, end))

    # generate plots and save
    ## actual vs predicted
    ### just x
    plt.plot(x_range, test_data[start:end, 0], color="black", label="Actual")
    plt.plot(x_range, preds[start:end, 0], color="black", linestyle="dotted", label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Trajectory of L-63 (N={})".format(N))
    plt.xlabel("Lyapunov Time")
    plt.ylabel("x")
    plt.savefig(dir + "actual_vs_predicted_x.pdf", bbox_inches="tight")
    plt.close()
    
    ### just y
    plt.plot(x_range, test_data[:end, 1], color="black", label="Actual")
    plt.plot(x_range, preds[:end, 1], color="black", linestyle="dotted", label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Trajectory of L-63 (N={})".format(N))
    plt.xlabel("Lyapunov Time")
    plt.ylabel("y")
    plt.savefig(dir + "actual_vs_predicted_y.pdf", bbox_inches="tight")
    plt.close()

    ### just z
    plt.plot(x_range, test_data[:end, 2], color="black", label="Actual")
    plt.plot(x_range, preds[:end, 2], color="black", linestyle="dotted", label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Trajectory of L-63 (N={})".format(N))
    plt.xlabel("Lyapunov Time")
    plt.ylabel("z")
    plt.savefig(dir + "actual_vs_predicted_z.pdf", bbox_inches="tight")
    plt.close()

    ## actual x-z
    plt.plot(test_data[start:end, 0], test_data[start:end, 2])
    plt.title("Actual Trajectory of L-63 in x-z space (N={})".format(N))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig(dir + "actual_x_z.pdf", bbox_inches="tight")
    plt.close()

    ## predicted x-z
    plt.plot(preds[start:end, 0], preds[start:end, 2])
    plt.title("Predicted Trajectory of L-63 in x-z space (N={})".format(N))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig(dir + "predicted_x_z.pdf", bbox_inches="tight")
    plt.close()

    ## actual vs predicted x-z
    plt.plot(test_data[start:end, 0], test_data[start:end, 2], color="black", label="Actual")
    plt.plot(preds[start:end, 0], preds[start:end, 2], color="red", linestyle="dotted", label="Predicted")
    plt.title("Actual vs Predicted Trajectory of L-63 in x-z space (N={})".format(N))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.savefig(dir + "actual_vs_predicted_x_z.pdf", bbox_inches="tight")
    plt.close()

    # write losses
    rmse = losses[0]
    griffith_rmse = losses[1]
    griffith_epsilon = error.griffith_epsilon(test_data, preds, DEL_T, LAMBDA)

    with open(dir + "losses.txt", 'w') as f:
        f.write("rmse: {}\ngriffith_rmse: {}\ngriffith_epsilon: {}\n".format(rmse, griffith_rmse, griffith_epsilon))

    print("Completed #{} / 500".format(trial_num))

if __name__ == '__main__':
    print("Beginning write.")
    start = datetime.now()

    counter = 0
    total = len(N_RANGE) * len(TRIAL_RANGE)

    paramgrid = itertools.product(N_RANGE, TRIAL_RANGE)
    pool = multiprocessing.Pool()
    result = pool.starmap(write_trial_data, paramgrid)
    pool.close()
    pool.join()

    end = datetime.now()
    computation_time = (end - start).total_seconds() / 60
    print("Write complete.")
    print("Time taken: {} mins".format(computation_time))