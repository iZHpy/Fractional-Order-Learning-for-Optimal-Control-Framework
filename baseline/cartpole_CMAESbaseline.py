import numpy as np
import pandas as pd
import time
import cma  # CMA-ES package
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='CFNO')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--N', type=int, default=8, help='Prediction horizon for MPC')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Cart-pole dynamics function
def cartpole_dynamics(state, u, params):
    x, x_dot, theta, theta_dot = state
    mc, mp, l, g = params

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    theta_ddot = (
        g * sin_theta + cos_theta * (-mp * l * theta_dot**2 * sin_theta - u) / (mc + mp)
    ) / (l * (4/3 - mp * cos_theta**2 / (mc + mp)))
    
    x_ddot = (
        mp * l * (theta_dot**2 * sin_theta - theta_ddot * cos_theta) + u
    ) / (mc + mp)
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

# Simulate the system using a given control sequence
def simulate_with_control(initial_state, u_sequence, params, dt):
    state = np.array(initial_state)
    trajectory = [state]
    for u in u_sequence:
        state_dot = cartpole_dynamics(state, u, params)
        state = state + dt * state_dot
        trajectory.append(state)
    return np.array(trajectory)

# Cost function for a finite horizon (with terminal cost)
def horizon_cost(u_sequence, initial_state, Q, R, params, dt, horizon):
    u_sequence = u_sequence.reshape(-1, 1)
    state = np.array(initial_state)
    cost = 0
    for i in range(horizon):
        u = u_sequence[i, 0]
        cost += state.T @ Q @ state + u * R * u  # running cost
        state_dot = cartpole_dynamics(state, u, params)
        state = state + dt * state_dot
    cost += state.T @ Q @ state  # terminal cost
    return cost

# Solve the finite horizon optimization using CMA-ES
def solve_mpc_finite_horizon(initial_state, Q, R, horizon, params, dt, u_bounds=(-10, 10)):
    u0 = np.zeros(horizon)  # initial guess for the control sequence
    sigma0 = 0.3  # initial standard deviation for CMA-ES
    # Set up bounds for each control input
    lower_bounds = [u_bounds[0]] * horizon
    upper_bounds = [u_bounds[1]] * horizon
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'verb_log': 0,   # disable CMA-ES logging
        'popsize': 20,   # population size (adjust as needed)
        'maxiter': 100   # maximum iterations (adjust as needed)
    }
    # Run CMA-ES
    res = cma.fmin(
        horizon_cost,
        u0,
        sigma0,
        args=(initial_state, Q, R, params, dt, horizon),
        options=options
    )
    best_control_sequence = res[0]
    return best_control_sequence

# Receding horizon MPC simulation: at each time step, solve for the optimal control sequence 
# over the finite horizon using CMA-ES, apply the first control, and update the state.
def simulate_mpc(initial_state, Q, R, horizon, sim_time, params, dt, u_bounds=(-10, 10)):
    state = np.array(initial_state)
    trajectory = [state]
    controls = []
    steps = int(sim_time / dt)
    for _ in range(steps):
        # Solve for the optimal control sequence over the finite horizon using CMA-ES
        u_opt = solve_mpc_finite_horizon(state, Q, R, horizon, params, dt, u_bounds)
        u = u_opt[0]  # apply only the first control action
        controls.append(u)
        # Update the state with the chosen control
        state_dot = cartpole_dynamics(state, u, params)
        state = state + dt * state_dot
        trajectory.append(state)
    return np.array(trajectory), np.array(controls)

# Load data for trajectories, optimal controls, and cost matrices
args = parse_args()
set_seed(args.seed)
dir_path = './data/cartpole/'
trajectories = np.load(dir_path+'fractional_system_trajectories.npy')
U_optimal = np.load(dir_path+'optimal_control_U.npy')
Q_matrices = np.load(dir_path+'LQR_Q.npy')
R_matrices = np.load(dir_path+'LQR_R.npy')

# System parameters
mc, mp, l, g = 1.0, 0.1, 1.0, 9.81
params = (mc, mp, l, g)
dt = 0.1
time_horizon = 16
sim_time = 3.2
u_bounds = (-0.5, 0.5)

mses = []
maes = []

# Record the start time
start_time = time.time()
test_indice = train_test_split(np.arange(trajectories.shape[0]), test_size=0.2, random_state=args.seed)[1]

for i in tqdm(test_indice):
    # Initial state
    x0 = trajectories[i, 0]
    
    # Cost matrices for this sample
    Q = Q_matrices[i]
    R = R_matrices[i]
    
    trajectory_est, estimated_controls_est = simulate_mpc(x0, Q, R, time_horizon, sim_time, params, dt, u_bounds)
    
    mae = mean_absolute_error(estimated_controls_est, U_optimal[i])
    mse = mean_squared_error(estimated_controls_est, U_optimal[i])
    print("Iteration", i, "MAE:", mae, "MSE:", mse)
    maes.append(mae)
    mses.append(mse)

# Record the end time and print total runtime
end_time = time.time()
total_runtime = end_time - start_time
print("Total runtime: {:.2f} seconds".format(total_runtime))
print("Mean MAE:", np.mean(maes))
print("Mean MSE:", np.mean(mses))