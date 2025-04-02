import casadi as cs
import numpy as np
import cma  # CMA-ES package
import matplotlib.pyplot as plt
import time
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


# Quadrotor class with provided dynamics
class Quadrotor:
    def __init__(self, quad_params):
        self.quad = quad_params
        self.x = cs.MX.sym('x', 13)  # State: [p, q, v, r]
        self.u = cs.MX.sym('u', 4)   # Control inputs

    def quad_dynamics(self):
        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
        return cs.Function('x_dot', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        return self.x[6:9]  # Velocity part

    def q_dynamics(self):
        return 0.5 * cs.mtimes(skew_symmetric(self.x[9:12]), self.x[3:7])

    def v_dynamics(self):
        f_thrust = self.u * self.quad['max_thrust']
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, cs.sum1(f_thrust)) / self.quad['mass']
        v_dynamics = v_dot_q(a_thrust, self.x[3:7]) - g
        return v_dynamics

    def w_dynamics(self):
        f_thrust = self.u * self.quad['max_thrust']
        y_f = cs.MX(self.quad['y_f'])
        x_f = cs.MX(self.quad['x_f'])
        c_f = cs.MX(self.quad['z_l_tau'])
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.quad['J'][1] - self.quad['J'][2]) * self.x[11] * self.x[12]) / self.quad['J'][0],
            (-cs.mtimes(f_thrust.T, x_f) + (self.quad['J'][2] - self.quad['J'][0]) * self.x[12] * self.x[10]) / self.quad['J'][1],
            (cs.mtimes(f_thrust.T, c_f) + (self.quad['J'][0] - self.quad['J'][1]) * self.x[10] * self.x[11]) / self.quad['J'][2]
        )

# Provided helper functions
def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)
    return cs.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    if isinstance(q, np.ndarray):
        return np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)]
        ])
    return cs.vertcat(
        cs.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2))
    )

def skew_symmetric(v):
    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])
    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0)
    )

quad_params = {
    'mass': 1.0,
    'max_thrust': 5.0,
    'y_f': np.array([0.1, -0.1, -0.1, 0.1]),  # Four elements
    'x_f': np.array([0.1, 0.1, -0.1, -0.1]),   # Four elements
    'z_l_tau': np.array([-0.05, 0.05, -0.05, 0.05]),  # Four elements
    'J': np.array([0.01, 0.01, 0.02])
}

# Create a Quadrotor object and get the dynamics function
quad = Quadrotor(quad_params)
quad_dyn = quad.quad_dynamics()

# Simulation parameters
dt = 0.1   # Time step
T = 1.6    # Total simulation time
N = int(T / dt)  # Number of time steps

# Load precomputed data
args = parse_args()
set_seed(args.seed)
dir_path = './data/quadrotor/'
trajectories = np.load(dir_path+'fractional_system_trajectories.npy')
U_optimal = np.load(dir_path+'optimal_control_U.npy')
Q_matrices = np.load(dir_path+'LQR_Q.npy')
R_matrices = np.load(dir_path+'LQR_R.npy')

def horizon_cost(u, initial_state, Q, R, params, dt, time_horizon):
    """
    Compute the cost over the entire time horizon for MPC.
    """
    quad_instance = Quadrotor(params)
    quad_dyn = quad_instance.quad_dynamics()
    x = initial_state
    cost = 0
    for k in range(time_horizon):
        # Extract control input for time step k (4 inputs per step)
        uk = np.array(u[k * 4:(k + 1) * 4])
        cost += x.T @ Q @ x + uk.T @ R @ uk
        x_dot = np.array(quad_dyn(x=cs.DM(x), u=cs.DM(uk))['x_dot']).flatten()
        x = x + dt * x_dot
    cost += x.T @ Q @ x  # Terminal cost
    return cost

def solve_mpc_full_horizon(initial_state, Q, R, time_horizon, params, dt, u_bounds=(-10, 10)):
    """
    Solve the MPC problem for the full horizon using CMA-ES.
    Returns the entire control sequence (length: time_horizon*4).
    """
    u0 = np.zeros(time_horizon * 4)   # Initial guess
    sigma0 = 1.0                      # Initial standard deviation for CMA-ES
    lower_bounds = [u_bounds[0]] * (time_horizon * 4)
    upper_bounds = [u_bounds[1]] * (time_horizon * 4)
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'verb_log': 0,     # Disable logging
        'popsize': 20,     # Population size (adjust as needed)
        'maxiter': 100     # Maximum iterations (adjust as needed)
    }
    res = cma.fmin(horizon_cost, u0, sigma0,
                   args=(initial_state, Q, R, params, dt, time_horizon),
                   options=options)
    return res[0]

def solve_mpc_finite_horizon(initial_state, Q, R, horizon_length, params, dt, u_bounds=(-10, 10)):
    """
    Solve the finite-horizon MPC problem using CMA-ES and return the first control input.
    """
    u0 = np.zeros(horizon_length * 4)
    sigma0 = 0.3
    lower_bounds = [u_bounds[0]] * (horizon_length * 4)
    upper_bounds = [u_bounds[1]] * (horizon_length * 4)
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'verb_log': 0,
        'popsize': 20,
        'maxiter': 100
    }
    res = cma.fmin(horizon_cost, u0, sigma0,
                   args=(initial_state, Q, R, params, dt, horizon_length),
                   options=options)
    optimal_u = res[0][:4]  # Use only the first control input from the sequence
    return optimal_u

def run_mpc(initial_state, Q, R, horizon_length, params, dt, total_steps, u_bounds=(-10, 10)):
    """
    Run the MPC in a receding horizon fashion.
    At each time step, solve the finite-horizon problem using CMA-ES,
    apply the first control input, and update the state.
    """
    state = initial_state.copy()
    trajectory = [state]
    control_history = []
    for t in range(total_steps):
        # Obtain the first control input via the finite-horizon CMA-ES solver
        u_opt = solve_mpc_finite_horizon(state, Q, R, horizon_length, params, dt, u_bounds)
        control_history.append(u_opt)
        quad_instance = Quadrotor(params)
        quad_dyn = quad_instance.quad_dynamics()
        x_dot = np.array(quad_dyn(x=cs.DM(state), u=cs.DM(u_opt))['x_dot']).flatten()
        state = state + dt * x_dot
        trajectory.append(state)
    return trajectory, control_history

# Prepare lists to record errors
mses = []
maes = []
time_horizon = 8  # Prediction horizon (number of time steps)
u_bounds = (-1, 1)

start_time = time.time()
test_indice = train_test_split(np.arange(trajectories.shape[0]), test_size=0.2, random_state=args.seed)[1]

for i in tqdm(test_indice):
    x0 = trajectories[i, 0]
    Q = Q_matrices[i]
    R = R_matrices[i]
    
    trajectory_est, estimated_controls_est = run_mpc(x0, Q, R, time_horizon, quad_params, dt, N, u_bounds)
    
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