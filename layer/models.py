import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.layers import ResidualMLPBlock

def get_config(config):
    num_blocks = config['identification_model_params'].get('num_blocks', 2)
    block_dims = config['identification_model_params'].get('hidden_size', [64, 64])
    block_layers = config['identification_model_params'].get('block_layers', [2, 2])
    dropout_rate = config['identification_model_params'].get('dropout_rate', 0.0)
    norm_type = config['identification_model_params'].get('norm_type', None)
    activation = config['identification_model_params'].get('activation', 'relu')
    use_residual = config['identification_model_params'].get('use_residual', False)
    return num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual


class MLPParamRegressor(nn.Module):
    def __init__(self,
                 n, m, T,
                 config
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        
        # input_dim: x0 + U
        in_dim = n + m*T
        # output_dim: A(n*n) + B(n*m) + alpha(n)
        out_dim = n*n + n*m + n
           
        num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual = get_config(config)

        # Stack muti ResidualMLPBlock
        blocks = []
        prev_dim = in_dim
        for i in range(num_blocks):
            block = ResidualMLPBlock(
                in_dim=prev_dim,
                hidden_dims=block_dims[i],
                num_layers=block_layers[i],
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                activation=activation,
                use_residual=use_residual  
            )
            blocks.append(block)
            prev_dim = block_dims[i][-1]
        self.blocks = nn.Sequential(*blocks)

        # Lasy linear layer: hidden_dim -> out_dim
        self.output_proj = nn.Linear(prev_dim, out_dim)

    def forward(self, x0, U):
        x0_flat = x0.view(x0.size(0), -1)
        U_flat = U.view(U.size(0), -1)
        inp = torch.cat([x0_flat, U_flat], dim=1)  # shape=(batch_size, in_dim), 

        # ResidualMLPBlocks
        x = self.blocks(inp)        # (batch_size, hidden_dim)

        # Last linear layer
        out = self.output_proj(x) # (batch_size, out_dim)

        # divide the output into A, B, alpha
        nn_ = self.n*self.n
        nm_ = self.n*self.m
        A_flat = out[:, :nn_]
        B_flat = out[:, nn_ : nn_ + nm_]
        alpha_flat = out[:, nn_ + nm_:]
        alpha = alpha_flat

        return A_flat, B_flat, alpha
    
# Define the CFNO model 
class CFNO(nn.Module):
    def __init__(self,
                 n, m, T,
                 config, logger
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config
        if config['identification_model'] == 'MLP':
            self.param_regressor = MLPParamRegressor(n, m, T, config)
        elif config['identification_model'] == 'RNN':
            logger.error(f"Identification model {config['identification_model']} not implemented")
            raise NotImplementedError(f"Identification model {config['identification_model']} not implemented")
        elif config['identification_model'] == 'LSTM':
            logger.error(f"Identification model {config['identification_model']} not implemented")
            raise NotImplementedError(f"Identification model {config['identification_model']} not implemented")
    
    
    def _get_batch(self, batch):
        x0 = batch['input_x']
        U = batch['input_u']
        A_true = batch['A']
        B_true = batch['B']
        alpha_true = batch['alphas']
        LQR_Q = batch['LQR_Q']
        LAR_R = batch['LQR_R']
        optimal_controls = batch['optimal_controls']
        return x0, U, A_true, B_true, alpha_true, LQR_Q, LAR_R, optimal_controls
    
    def _get_regress_loss(self, A, B, alpha, A_true, B_true, alpha_true, config, logger):
        loss_type = config['identification_model_params'].get('loss', 'MSE')
        if loss_type == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
            A_true = A_true.view(A.shape[0], -1)
            B_true = B_true.view(B.shape[0], -1)
        else:
            logger.error(f"Loss type {loss_type} not implemented")
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        
        loss_A = criterion(A, A_true)
        loss_B = criterion(B, B_true)
        loss_alpha = criterion(alpha, alpha_true)

        reduction = config['identification_model_params']['loss_params'].get('reduction', 'sum')
        if reduction == 'mean':
            return torch.mean(torch.stack([loss_A, loss_B, loss_alpha]))
        elif reduction == 'sum':
            return torch.sum(torch.stack([loss_A, loss_B, loss_alpha]))
        else:
            logger.error(f"Reduction type {reduction} not implemented")
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    

    def forward(self, batch, logger):
        x0, U, A_true, B_true, alpha_true, LQR_Q, LAR_R, optimal_controls = self._get_batch(batch)
        A, B, alpha = self.param_regressor(x0, U)
        regress_loss = self._get_regress_loss(A, B, alpha, A_true, B_true, alpha_true, self.cfg, logger)
        return A, B, alpha