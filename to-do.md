# General
- [ ] Implement Griffith, Pomerance & Gauthier (2019) - Forecasting Chaotic Systems
with Very Low Connectivity Reservoir Computers
- [ ] Modify `data.generate_**()` functions washout code. Either according to 
Griffith (2019) or moving spatial average (to ensure it has converged on the
attractor)

# Check Performance and Optimisation of Hyperparameters
- [ ] Train on an IC -> test using just that IC -> RMSE (a bit higher)
- [ ] Train on an IC -> test using another IC -> RMSE (a bit lower)

Are the above in the same order of magnitude as Griffith (2019)?

# Computational

## Distribution of Testing Error
For each, keep the hyperparameters fixed. But the random matrices A, W_in are 
drawn each iteration. 

For each iteration, store each as a file
- IC
- Actual Trajectory
- A and W_in
- Reservoir
- Predicted Trajectory
- RMSE(actual_trajectory, predicted_trajectory)

Checklist:
- [ ] (Ordinary) Train on an IC -> test using 1000 other IC's (new A, W_in) -> RMSE
- [ ] (Just x) Train on an IC -> test using 1000 other IC's (new A, W_in) -> RMSE
- [ ] (Delay) Train on an IC -> test using 1000 other IC's (new A, W_in) -> RMSE

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

# Theoretical
Assume the activation function is linear (replace $\tanh$ with identity). 
- [ ] Write out some terms for the reservoir.
- [ ] Is there any discernible pattern? Can any conclusions be drawn?
- [ ] If so, try to validate the claims empirically. 