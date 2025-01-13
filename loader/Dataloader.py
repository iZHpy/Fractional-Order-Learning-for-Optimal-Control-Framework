import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FractionalOrderDataset(Dataset):
    def __init__(self, alphas, inputs, x_init, LQR_Q, LQR_R, optimal_controls, A, B, config):
        self.data = {
            'alphas': torch.tensor(alphas, dtype=torch.float32),
            'input_u': torch.tensor(inputs, dtype=torch.float32),
            'input_x': torch.tensor(x_init if config['x_mode'] == 'All' else x_init[:, :1], dtype=torch.float32),
            'LQR_Q': torch.tensor(LQR_Q, dtype=torch.float32),
            'LQR_R': torch.tensor(LQR_R, dtype=torch.float32),
            'optimal_controls': torch.tensor(optimal_controls, dtype=torch.float32),
            'A': torch.tensor(A, dtype=torch.float32),
            'B': torch.tensor(B, dtype=torch.float32)
        }
        self.m = inputs.shape[2]
        self.n = x_init.shape[2]
        self.T = inputs.shape[1]
        self.config = config
        self.len = inputs.shape[0]
        self.data['A'] = self._duplicate_and_stack(self.data['A'], self.len)
        self.data['B'] = self._duplicate_and_stack(self.data['B'], self.len)
        self.data['alphas'] = self._duplicate_and_stack(self.data['alphas'], self.len)

    def _duplicate_and_stack(self, tensor, num_copies):
        return torch.stack([tensor] * num_copies, dim=0)


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.data[idx]
        else:
            return {key: value[idx] for key, value in self.data.items()}

    
def load_data_from_npy(data_dir):
    alphas = np.load(f'{data_dir}/fractional_orders_alpha.npy')
    inputs = np.load(f'{data_dir}/fractional_system_inputs.npy')
    x_init = np.load(f'{data_dir}/fractional_system_trajectories.npy')
    LQR_Q = np.load(f'{data_dir}/LQR_Q.npy')
    LQR_R = np.load(f'{data_dir}/LQR_R.npy')
    optimal_controls = np.load(f'{data_dir}/optimal_control_U.npy')
    A = np.load(f'{data_dir}/system_matrix_A.npy')
    B = np.load(f'{data_dir}/system_matrix_B.npy')
    
    return alphas, inputs, x_init, LQR_Q, LQR_R, optimal_controls, A, B

# Split data into training and testing sets
def load_split_data(config):
    data_dir = config['data_dir']
    test_size = config['test_size']
    random_state = config['seed']
    alphas, inputs, x_init, LQR_Q, LQR_R, optimal_controls, A, B = load_data_from_npy(data_dir)
    train_indices, test_indices = train_test_split(np.arange(inputs.shape[0]), test_size=test_size, random_state=random_state)
    
    train_dataset = FractionalOrderDataset(alphas, inputs[train_indices], x_init[train_indices], LQR_Q[train_indices], LQR_R[train_indices], optimal_controls[train_indices], A, B, config)
    test_dataset = FractionalOrderDataset(alphas, inputs[test_indices], x_init[test_indices], LQR_Q[test_indices], LQR_R[test_indices], optimal_controls[test_indices], A, B, config)
    
    return train_dataset, test_dataset


