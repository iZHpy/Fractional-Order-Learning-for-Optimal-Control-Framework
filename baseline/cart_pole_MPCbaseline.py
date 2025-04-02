import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
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

# Solve the MPC optimization over a finite horizon
def solve_mpc_finite_horizon(initial_state, Q, R, horizon, params, dt, u_bounds=(-10, 10)):
    u0 = np.zeros(horizon)  # initial guess for the control sequence
    bounds = [(u_bounds[0], u_bounds[1])] * horizon
    result = minimize(
        horizon_cost, u0, args=(initial_state, Q, R, params, dt, horizon),
        bounds=bounds, method='SLSQP'
    )
    return result.x

# Receding horizon MPC simulation: at each time step, solve for the optimal control sequence 
# over the finite horizon, apply the first control, and update the state.
def simulate_mpc(initial_state, Q, R, horizon, sim_time, params, dt, u_bounds=(-10, 10)):
    state = np.array(initial_state)
    trajectory = [state]
    controls = []
    steps = int(sim_time / dt)
    for _ in range(steps):
        # Solve for the optimal control sequence over the finite horizon
        u_opt = solve_mpc_finite_horizon(state, Q, R, horizon, params, dt, u_bounds)
        u = u_opt[0]  # apply only the first control action
        controls.append(u)
        # Update the state with the chosen control
        state_dot = cartpole_dynamics(state, u, params)
        state = state + dt * state_dot
        trajectory.append(state)
    return np.array(trajectory), np.array(controls)


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
#T = 32
sim_time = 3.2
num_samples = 20000
u_bounds = (-0.5, 0.5) 


mses = []
maes = []

# Record the start time
start_time = time.time()
test_indice = train_test_split(np.arange(trajectories.shape[0]), test_size=0.2, random_state=args.seed)[1]


for i in tqdm(test_indice):
    # Initial state
    x0 = trajectories[i,0]

    # Cost matrices
    Q = Q_matrices[i]
    R = R_matrices[i]

    trajectory_est, estimated_controls_est = simulate_mpc(x0, Q, R, time_horizon, sim_time, params, dt, u_bounds)
    

    mae = mean_absolute_error(estimated_controls_est, U_optimal[i])
    mse = mean_squared_error(estimated_controls_est, U_optimal[i])
    print("Iteration", i, "MAE:", mae, "MSE:", mse)
    maes.append(mae)
    mses.append(mse)

# Record the end time
end_time = time.time()

# Calculate and print the overall runtime
total_runtime = end_time - start_time
print("Total runtime: {:.2f} seconds".format(total_runtime))
print("Average MAE:", np.mean(maes))
print("Average MSE:", np.mean(mses))