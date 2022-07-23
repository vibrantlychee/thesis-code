# Programming Component
`data.py`
- [x] Modify `data.generate_**()` functions washout code. Either according to 
Griffith (2019) or moving spatial average (to ensure it has converged on the
attractor)

`error.py`
- [ ] implement forecast time

`reservoir.py`
Implement Griffith, Pomerance & Gauthier (2019) - Forecasting Chaotic Systems
with Very Low Connectivity Reservoir Computers:
- [x] Internal reservoir construction
- [x] Reservoir dynamics
- [x] Echo state network generation (training)
- [x] Regularised regression (Tikhonov)
- [x] Echo state network generation (test)
- [x] Forecasting
- [ ] Run a test to see if bayesian optimisation for every trial is 
computationally feasible

## Optimisation
~~Why might the optimisation not be working?~~
- ~~Perform grid search around the given optimum~~
- ~~Check if more nodes helps~~
- ~~Check if longer training period helps~~
Reservoir computer now works

## Check Performance and Optimisation of Hyperparameters
- [x] Train on an IC -> test using just that IC -> RMSE (a bit higher)
- [x] Train on an IC -> test using another IC -> RMSE (a bit lower)

**Are the above in the same order of magnitude as Griffith (2019)?**
Yes!

## Packaging into PyPi
- [ ] Exception handling.
- [ ] Learn how to package python projects.

# Computational Component

## Distribution of Testing Error
For each trial, keep the hyperparameters fixed, and the random matrices $W_r$,
$W_{\text{in}}$ are drawn each iteration. 

For each iteration, store each in the same `.npz` file:
- IC
- Actual Trajectory
- $W_r$ and $W_{\text{in}}$
- Reservoir
- Predicted Trajectory
- RMSE(actual_trajectory, predicted_trajectory)

Checklist:
- [ ] (Ordinary) Train on an IC -> test using 1000 other IC's (new $W_r$, $W_{\text{in}}$) -> RMSE
- [ ] (Just x) Train on an IC -> test using 1000 other IC's (new $W_r$, $W_{\text{in}}$) -> RMSE
- [ ] (Delay) Train on an IC -> test using 1000 other IC's (new $W_r$, $W_{\text{in}}$) -> RMSE

For Ordinary, Just x, and Delay, plot the histogram of RMSE's. 

## Distribution of Reservoir Values
For Ordinary, Just x, and Delay, do the following:
- [ ] Define good RMSE and bad RMSE
- [ ] Find `good_indices` and `bad_indices`.
- [ ] (Chart 1) On one chart, show the histogram of reservoir values for 
`good_indices`.
- [ ] (Chart 2) On one chart, show the histogram of reservoir values for 
`bad_indices`.

Do Chart 1 and Chart 2 resemble Fig.2 in GottwaldReich_PhysicaD2021?

## Random Feature Maps
- [ ] Implement
- [ ] Repeat above

## Roessler
- [ ] Repeat above for Roessler system

# Analytical Component
Assume the activation function is linear (replace $\tanh$ with identity). 
- [ ] Write out some terms for the reservoir.
- [ ] Is there any discernible pattern? Can any conclusions be drawn?
- [ ] If so, try to validate the claims empirically. 