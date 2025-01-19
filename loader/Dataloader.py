import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.utils import UnitGaussianNormalizer, GaussianNormalizer


class FractionalOrderDataset(Dataset):
    def __init__(self, alphas, input_u, x_init, LQR_Q, LQR_R, optimal_controls, A, B, config, normalize=True):
        self.data = {
            'alphas': torch.tensor(alphas, dtype=torch.float32),
            'input_u': torch.tensor(input_u, dtype=torch.float32),
            'input_x': torch.tensor(x_init[:,:-1] if config['x_mode'] == 'All' else x_init[:, :1], dtype=torch.float32),
            'LQR_Q': torch.tensor(LQR_Q, dtype=torch.float32),
            'LQR_R': torch.tensor(LQR_R, dtype=torch.float32),
            'optimal_controls': torch.tensor(optimal_controls, dtype=torch.float32),
            'A': torch.tensor(A, dtype=torch.float32),
            'B': torch.tensor(B, dtype=torch.float32)
        }
        self.m = input_u.shape[2]
        self.n = x_init.shape[2]
        self.T = input_u.shape[1]
        self.config = config
        self.len = input_u.shape[0]
        if config['same_system'] == True:
            self.data['A'] = self._duplicate_and_stack(self.data['A'], self.len)
            self.data['B'] = self._duplicate_and_stack(self.data['B'], self.len)
            self.data['alphas'] = self._duplicate_and_stack(self.data['alphas'], self.len)

        # output (batch_size, t, dim)
        # x (batch_size, t', dim)

        if normalize:
            self._normalize_data()
        else:
            self.norms = None
                

    def _duplicate_and_stack(self, tensor, num_copies):
        return torch.stack([tensor] * num_copies, dim=0)

    def _normalize_data(self):
        norms = {}
        for key in self.data:
            norms[key] = UnitGaussianNormalizer(self.data[key])
            self.data[key] = norms[key].encode(self.data[key])
        self.norms = norms

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.data[idx]
        else:
            return {key: value[idx] for key, value in self.data.items()}

    
def load_data_from_npy(data_dir):
    alphas = np.load(f'{data_dir}/fractional_orders_alpha.npy')
    input_u = np.load(f'{data_dir}/fractional_system_inputs.npy')
    x = np.load(f'{data_dir}/fractional_system_trajectories.npy')
    LQR_Q = np.load(f'{data_dir}/LQR_Q.npy')
    LQR_R = np.load(f'{data_dir}/LQR_R.npy')
    optimal_controls = np.load(f'{data_dir}/optimal_control_U.npy')
    A = np.load(f'{data_dir}/system_matrix_A.npy')
    B = np.load(f'{data_dir}/system_matrix_B.npy')
    print('álphas shape:', alphas.shape)
    print('input_u shape:', input_u.shape)
    print('x shape:', x.shape)
    print('LQR_Q shape:', LQR_Q.shape)
    print('LQR_R shape:', LQR_R.shape)
    print('optimal_controls shape:', optimal_controls.shape)
    print('A shape:', A.shape)
    print('B shape:', B.shape)

    return alphas, input_u, x, LQR_Q, LQR_R, optimal_controls, A, B

# Split data into training and testing sets
def load_split_data(config):
    data_dir = config['data_dir']
    test_size = config['test_size']
    random_state = config['seed']
    normalize = config['normalize']
    alphas, inputs, x, LQR_Q, LQR_R, optimal_controls, A, B = load_data_from_npy(data_dir)
    train_indices, test_indices = train_test_split(np.arange(inputs.shape[0]), test_size=test_size, random_state=random_state)
    
    train_dataset = FractionalOrderDataset(alphas, inputs[train_indices], x[train_indices], LQR_Q[train_indices], LQR_R[train_indices], optimal_controls[train_indices], A, B, config, normalize=normalize)
    test_dataset = FractionalOrderDataset(alphas, inputs[test_indices], x[test_indices], LQR_Q[test_indices], LQR_R[test_indices], optimal_controls[test_indices], A, B, config, normalize=normalize)
    train_norms = train_dataset.norms
    test_norms = test_dataset.norms

    return train_dataset, test_dataset, train_norms, test_norms


