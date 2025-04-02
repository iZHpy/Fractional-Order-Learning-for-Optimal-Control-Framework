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

# Simulate the system using a PID controller
def simulate_pid(initial_state, pid_gains, sim_time, params, dt, desired_state=None, u_bounds=(-10, 10)):
    if desired_state is None:
         desired_state = np.zeros_like(initial_state)
    state = np.array(initial_state)
    trajectory = [state]
    controls = []
    error_integral = np.zeros_like(state)
    previous_error = desired_state - state
    steps = int(sim_time / dt)
    for i in range(steps):
         # Compute the error between the desired state and current state
         error = desired_state - state
         error_integral += error * dt
         error_derivative = (error - previous_error) / dt
         previous_error = error
         
         # Compute the control action using the PID law:
         # u = Kp*error + Ki*integral(error) + Kd*derivative(error)
         # Here we use vector gains and combine the four state errors into a single control input.
         u = np.dot(pid_gains['Kp'], error) + np.dot(pid_gains['Ki'], error_integral) + np.dot(pid_gains['Kd'], error_derivative)
         u = np.clip(u, u_bounds[0], u_bounds[1])
         controls.append(u)
         
         # Update the state using Euler integration
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

mc, mp, l, g = 1.0, 0.1, 1.0, 9.81
params = (mc, mp, l, g)
dt = 0.1
sim_time = 3.2
u_bounds = (-0.5, 0.5) 

# Define PID gains.
# These gains represent the proportional, integral, and derivative contributions from each state error.
# In many cart-pole systems, the pole angle (theta) is critical so it is given a higher weight.
pid_gains = {
    'Kp': np.array([1.0, 0.0, 15.0, 0.0]),
    'Ki': np.array([0.1, 0.0, 1.0, 0.0]),
    'Kd': np.array([0.5, 0.0, 3.0, 0.0])
}

maes = []
mses = []

# Record the start time
start_time = time.time()
test_indice = train_test_split(np.arange(trajectories.shape[0]), test_size=0.2, random_state=args.seed)[1]

for i in tqdm(test_indice):
    # Get the initial state from the preloaded trajectory data
    x0 = trajectories[i, 0]
    
    # Simulate using the PID controller.
    # Here we set the desired state to zero.
    trajectory_est, estimated_controls_est = simulate_pid(x0, pid_gains, sim_time, params, dt, desired_state=np.zeros(4), u_bounds=u_bounds)
    
    # Compare the PID control actions to the pre-computed optimal control (from LQR/MPC)
    mae = mean_absolute_error(estimated_controls_est, U_optimal[i])
    mse = mean_squared_error(estimated_controls_est, U_optimal[i])
    print("Iteration", i, "MAE:", mae, "MSE:", mse)
    maes.append(mae)
    mses.append(mse)

end_time = time.time()
total_runtime = end_time - start_time
print("Total runtime: {:.2f} seconds".format(total_runtime))
print("Mean MAE:", np.mean(maes))
print("Mean MSE:", np.mean(mses))
