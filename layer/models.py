import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.layers import ResidualMLPBlock

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
           
        num_blocks = config['identification_model_params'].get('num_blocks', 2)
        block_dims = config['identification_model_params'].get('hidden_dims', [64, 64])
        block_layers = config['identification_model_params'].get('block_layers', [2, 2])
        dropout_rate = config['identification_model_params'].get('dropout_rate', 0.0)
        norm_type = config['identification_model_params'].get('norm_type', None)
        activation = config['identification_model_params'].get('activation', 'relu')
        use_residual = config['identification_model_params'].get('use_residual', False)

        # Stack muti ResidualMLPBlock
        blocks = []
        prev_dim = in_dim
        for i in range(num_blocks):
            block = ResidualMLPBlock(
                in_dim=prev_dim,
                hidden_dim=block_dims[i],
                num_layers=block_layers[2],
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
        x0_flat = x0.view(-1)
        U_flat = U.view(-1)
        inp = torch.cat([x0_flat, U_flat], dim=0).unsqueeze(0)  # shape=(1, in_dim), batch_size=1


        # ResidualMLPBlocks
        x = self.blocks(x)        # (1, hidden_dim)

        # 输出投影
        out = self.output_proj(x) # (1, out_dim)
        out = out.squeeze(0)      # (out_dim,)

        # 拆分成 A, B, alpha
        nn_ = self.n*self.n
        nm_ = self.n*self.m
        A_flat = out[:nn_]
        B_flat = out[nn_ : nn_ + nm_]
        alpha_flat = out[nn_ + nm_:]
        A = A_flat.view(self.n, self.n)
        B = B_flat.view(self.n, self.m)
        alpha = alpha_flat

        return A, B, alpha
    
class CFNO(nn.Module):
    def __init__(self,
                 n, m, T,
                 config
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        
        self.param_regressor = MLPParamRegressor(n, m, T, config) if config['identification_model'] == 'MLP' else None

    def forward(self, x0, U):
        A, B, alpha = self.param_regressor(x0, U)
        return A, B, alpha