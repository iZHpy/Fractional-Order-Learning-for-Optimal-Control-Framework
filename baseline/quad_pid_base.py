import casadi as cs
import numpy as np
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
        # The dynamics are grouped into position, orientation, translational, and rotational dynamics
        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
        return cs.Function('x_dot', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        # p dynamics: the time derivative of the position is (a portion of) the velocity
        return self.x[6:9]  # Note: the indexing follows the provided formulation

    def q_dynamics(self):
        # q dynamics: quaternion derivative computed from the angular velocity
        return 0.5 * cs.mtimes(skew_symmetric(self.x[9:12]), self.x[3:7])

    def v_dynamics(self):
        # v dynamics: acceleration from thrust and gravity
        f_thrust = self.u * self.quad['max_thrust']
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, cs.sum1(f_thrust)) / self.quad['mass']
        return v_dot_q(a_thrust, self.x[3:7]) - g

    def w_dynamics(self):
        # w dynamics: angular acceleration computed from moments and gyroscopic effects
        f_thrust = self.u * self.quad['max_thrust']
        y_f = cs.MX(self.quad['y_f'])
        x_f = cs.MX(self.quad['x_f'])
        c_f = cs.MX(self.quad['z_l_tau'])
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.quad['J'][1] - self.quad['J'][2]) * self.x[11] * self.x[12]) / self.quad['J'][0],
            (-cs.mtimes(f_thrust.T, x_f) + (self.quad['J'][2] - self.quad['J'][0]) * self.x[12] * self.x[10]) / self.quad['J'][1],
            (cs.mtimes(f_thrust.T, c_f) + (self.quad['J'][0] - self.quad['J'][1]) * self.x[10] * self.x[11]) / self.quad['J'][2]
        )

# Helper functions
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
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)
    """
    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])
    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

# Quadrotor parameters
quad_params = {
    'mass': 1.0,
    'max_thrust': 5.0,
    'y_f': np.array([0.1, -0.1, -0.1, 0.1]),   # Four elements
    'x_f': np.array([0.1, 0.1, -0.1, -0.1]),     # Four elements
    'z_l_tau': np.array([-0.05, 0.05, -0.05, 0.05]),  # Four elements
    'J': np.array([0.01, 0.01, 0.02])
}

# Create a Quadrotor object (for dynamics evaluation)
quad = Quadrotor(quad_params)
quad_dyn = quad.quad_dynamics()

# Simulation parameters
dt = 0.1       # Time step
T = 1.6        # Total simulation time
N = int(T / dt)  # Number of time steps

# Load precomputed trajectories, optimal control inputs, and cost matrices (for evaluation)
args = parse_args()
set_seed(args.seed)
dir_path = './data/quadrotor/'
trajectories = np.load(dir_path+'fractional_system_trajectories.npy')
U_optimal = np.load(dir_path+'optimal_control_U.npy')
Q_matrices = np.load(dir_path+'LQR_Q.npy')
R_matrices = np.load(dir_path+'LQR_R.npy')

# Define the PID controller simulation for the quadrotor.
def simulate_pid_quad(initial_state, Kp, Ki, Kd, desired_state, quad_params, dt, total_steps, u_bounds=(-20, 20)):
    """
    Simulate the quadrotor using a PID controller.
    """
    state = initial_state.copy()
    trajectory = [state]
    control_history = []
    integral_error = np.zeros_like(state)
    previous_error = desired_state - state

    # Create a quadrotor object for dynamics evaluation.
    quad = Quadrotor(quad_params)
    quad_dyn = quad.quad_dynamics()

    for t in range(total_steps):
        # Compute error between desired state and current state
        error = desired_state - state
        integral_error = integral_error + error * dt
        derivative_error = (error - previous_error) / dt
        previous_error = error

        # Compute the control action using the PID law:
        # u = Kp*error + Ki*integral_error + Kd*derivative_error
        u = Kp.dot(error) + Ki.dot(integral_error) + Kd.dot(derivative_error)
        u = np.clip(u, u_bounds[0], u_bounds[1])
        control_history.append(u)

        # Compute state derivative using quadrotor dynamics (evaluated via CasADi)
        x_dot = np.array(quad_dyn(x=cs.DM(state), u=cs.DM(u))['x_dot']).flatten()
        state = state + dt * x_dot
        trajectory.append(state)

    return np.array(trajectory), np.array(control_history)



# --- PID Gain Tuning ---
Kp = 0.1 * np.ones((4, 13))
Ki = 0.01 * np.ones((4, 13))
Kd = 0.05 * np.ones((4, 13))

# Define the desired state (here set to the zero state)
desired_state = np.zeros(13)
u_bounds=(-1, 1)
# Arrays to record error metrics
mses = []
maes = []

# Record the start time
start_time = time.time()

test_indice = train_test_split(np.arange(trajectories.shape[0]), test_size=0.2, random_state=args.seed)[1]

for i in tqdm(test_indice):
    # Initial state from the preloaded trajectory data
    x0 = trajectories[i, 0]

    # Simulate using the PID controller
    trajectory_est, estimated_controls_est = simulate_pid_quad(x0, Kp, Ki, Kd, desired_state, quad_params, dt, N, u_bounds)
    
    # Compare the PID control actions to the pre-computed optimal control (from LQR/MPC)
    mae = mean_absolute_error(estimated_controls_est, U_optimal[i])
    mse = mean_squared_error(estimated_controls_est, U_optimal[i])
    print("Iteration", i, "MAE:", mae, "MSE:", mse)
    maes.append(mae)
    mses.append(mse)

# Record the end time
end_time = time.time()
total_runtime = end_time - start_time
print("Total runtime: {:.2f} seconds".format(total_runtime))
print("Average MAE:", np.mean(maes))
print("Average MSE:", np.mean(mses))