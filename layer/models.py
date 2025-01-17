import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO2d
from layer.layers import ResidualMLPBlock
from layer.layers import LSTMGenerator, RNNGenerator, GRUGenerator


class MLPParamRegressor(nn.Module):
    def __init__(self,
                 n, m, T,
                 config
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        
        # input_dim: x + U    U (batch_size, T, dim), X (batch_size, 1 , dim)
        in_dim = n + m*T   # T * dim + dim  (batch_size, T*dim + dim)
        # output_dim: A(n*n) + B(n*m) + alpha(n)     A (batch_size, n*n), B (batch_size, n*m), alpha (batch_size, n)
        out_dim = n*n + n*m + n

        num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual = self._get_config(config)

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

    def _get_config(self, config):
        num_blocks = config['identification_model_params'].get('num_blocks', 2)
        block_dims = config['identification_model_params'].get('hidden_size', [64, 64])
        block_layers = config['identification_model_params'].get('block_layers', [2, 2])
        dropout_rate = config['identification_model_params'].get('dropout_rate', 0.0)
        norm_type = config['identification_model_params'].get('norm_type', None)
        activation = config['identification_model_params'].get('activation', 'ReLU')
        use_residual = config['identification_model_params'].get('use_residual', False)
        return num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual


    def forward(self, x, U):
        x_flat = x.view(x.size(0), -1)
        U_flat = U.view(U.size(0), -1)
        inp = torch.cat([x_flat, U_flat], dim=1)  # shape=(batch_size, in_dim), 

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

        return A_flat, B_flat, alpha_flat
    
    def _get_config(self, config):
        num_blocks = config['identification_model_params'].get('num_blocks', 2)
        block_dims = config['identification_model_params'].get('hidden_size', [64, 64])
        block_layers = config['identification_model_params'].get('block_layers', [2, 2])
        dropout_rate = config['identification_model_params'].get('dropout_rate', 0.0)
        norm_type = config['identification_model_params'].get('norm_type', None)
        activation = config['identification_model_params'].get('activation', 'ReLU')
        use_residual = config['identification_model_params'].get('use_residual', False)
        return num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual

class A_ParamEmbedding(nn.Module):
    def __init__(self,
                 n, m, T,
                 config
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config
        hidden_size, dropout_rate, norm_type, activation, use_residual, use_U = self._get_config(config)
        self.output_size = config['AParam_model_params'].get('output_size', n)

        self.Embedding = nn.Embedding(T, hidden_size//2)
        self.fc1 = nn.Linear(n*2, hidden_size)
        self.fc2 = nn.Linear(n, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size, self.output_size)
        self.act = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()][['ReLU', 'GELU',  'Tanh', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(self.output_size), nn.LayerNorm(self.output_size)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()


    def forward(self, U, A_flat, alpha):
        # calculate the eigenvalues of A
        A0 = A_flat.view(A_flat.shape[0], self.n, self.n) - torch.diag_embed(alpha)
        eigenvalues, eigenvectors = torch.linalg.eig(A0) # (batch_size, n), (batch_size, n, n)
        eigenvalues_real = torch.view_as_real(eigenvalues)  # (batch_size, n, 2)
        eigenvalues_real = eigenvalues_real.view(eigenvalues_real.shape[0], -1)  # (batch_size, n * 2)

        A0_emb = self.act(self.fc1(eigenvalues_real)) # (batch_size, n)
        alpha_emb = self.act(self.fc2(alpha.unsqueeze(1).repeat(1, self.T - 1, 1))) # (batch_size, T, hidden_size//2)

        Ts = torch.arange(self.T-1).unsqueeze(0).repeat(A_flat.size(0), 1).to(A_flat.device)
        T_emb = self.Embedding(Ts) # (batch_size, T-1, hidden_size//2)
        T_emb = torch.cat([T_emb, alpha_emb], dim=2)  # (batch_size, T-1, hidden_size)
        T_emb = torch.cat([A0_emb.unsqueeze(1), T_emb], dim=1)  # (batch_size, T, hidden_size)
        emb = self.dropout(self.act(self.fc3(T_emb))).transpose(0, 1)  # (T, batch_size, output_size)
        if self.norm_type == 'BatchNorm':
            emb = self.norm(emb.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.norm_type == 'LayerNorm':
            emb = self.norm(emb)
        return emb
    
    def _get_config(self, config):
        hidden_size = config['AParam_model_params'].get('hidden_size', 64)
        dropout_rate = config['AParam_model_params'].get('dropout_rate', 0.0)
        norm_type = config['AParam_model_params'].get('norm_type', None)
        activation = config['AParam_model_params'].get('activation', 'ReLU')
        use_residual = config['AParam_model_params'].get('use_residual', False)
        use_U = config['AParam_model_params'].get('use_U', False)
        return  hidden_size, dropout_rate, norm_type, activation, use_residual, use_U
        
class A_ParamMLP(nn.Module):
    def __init__(self, n, T, config=None):
        super(A_ParamMLP, self).__init__()
        num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual, use_U= self._get_config(config)
        # input: n*n + n
        self.input_dim = n*n + n
        # output: T* output_size 
        self.output_size = config['AParam_model_params'].get('output_size', n)
        self.output_dim = T * self.output_size
        
        # Stack muti ResidualMLPBlock
        blocks = []
        prev_dim = self.input_dim
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
        self.output_proj = nn.Linear(prev_dim, self.output_dim)

    def _get_config(self, config):
        num_blocks = config['AParam_model_params'].get('num_blocks', 2)
        block_dims = config['AParam_model_params'].get('hidden_size', [64, 64])
        block_layers = config['AParam_model_params'].get('block_layers', [2, 2])
        dropout_rate = config['AParam_model_params'].get('dropout_rate', 0.0)
        norm_type = config['AParam_model_params'].get('norm_type', None)
        activation = config['AParam_model_params'].get('activation', 'ReLU')
        use_residual = config['AParam_model_params'].get('use_residual', False)
        use_U = config['AParam_model_params'].get('use_U', False)
        return num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual, use_U
        
    def forward(self, U, A_flat, alpha):
        """
        A: (batch_size, n*n)
        alpha: (batch_size, n)
        return: (batch_size, T, output_size)
        """
        batch_size = A_flat.size(0)  
           
        # concat [A_flat, alpha]
        x = torch.cat([A_flat, alpha], dim=1)  # (batch_size, n*n + n)
        # forward
        x = self.blocks(x)  # (batch_size, hidden_dim)
        x = self.output_proj(x)  # (batch_size, T*output_size)
        # reshape (batch_size, T, n)
        x = x.view(batch_size, -1, self.output_size)  

        return x.transpose(0, 1)  # (T, batch_size, n)


class GParamModel(nn.Module):
    def __init__(self, n, m, T, input_dim, 
                 config, logger, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.cfg = config
        self.n = n
        self.m = m
        self.T = T
        model_type, hidden_size, num_layers, dropout_rate, norm_type, activation = self._get_config(config)
        if model_type == 'RNN':
            self.seq_model = LSTMGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=n*n, dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'LSTM':
            self.seq_model = RNNGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=n*n, dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'GRU':
            self.seq_model = GRUGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=n*n, dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'Transformer':
            logger.error(f"Model type {model_type} not implemented")
            raise NotImplementedError(f"Model type {model_type} not implemented")
        elif model_type == 'Formula':
            logger.error(f"Model type {model_type} not implemented")
            raise NotImplementedError(f"Model type {model_type} not implemented")
    
        # self.seq_model 
    def _get_config(self, config):
        model_type = config['GParam_model']
        hidden_size = config['GParam_model_params'].get('hidden_size', 64)
        num_layers = config['GParam_model_params'].get('block_layers', 2)
        dropout_rate = config['GParam_model_params'].get('dropout_rate', 0.0)
        norm_type = config['GParam_model_params'].get('norm_type', None)
        activation = config['GParam_model_params'].get('activation', 'ReLU')
        return model_type, hidden_size, num_layers, dropout_rate, norm_type, activation

    def forward(self, x):
        """
        x: (T, batch_size, input_dim)
        return: (T, batch_size, n*n)
        """
        return self.seq_model(x)


class FinalTrans(nn.Module):
    def __init__(self, n, m, T, 
                 config, logger, *args, **kwargs):
        super(FinalTrans, self).__init__()
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config
        self.input_dim = n
        num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual = self._get_config(config)
        self.input_proj = nn.Linear(self.input_dim, block_dims[0][0])
        blocks = []
        prev_dim = block_dims[0][0]
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
        self.output_proj = nn.Linear(prev_dim, self.m)
    
    
    def _get_config(self, config):
        num_blocks = config['FINAL_model_params'].get('num_blocks', 2)
        block_dims = config['FINAL_model_params'].get('hidden_size', [64, 64])
        block_layers = config['FINAL_model_params'].get('block_layers', [2, 2])
        dropout_rate = config['FINAL_model_params'].get('dropout_rate', 0.0)
        norm_type = config['FINAL_model_params'].get('norm_type', None)
        activation = config['FINAL_model_params'].get('activation', 'ReLU')
        use_residual = config['FINAL_model_params'].get('use_residual', False)
        return num_blocks, block_dims, block_layers, dropout_rate, norm_type, activation, use_residual
        
    def forward(self, B, G, LQR_Q, LQR_R, x):
        """
        B: (batch_size, n*m)
        G: (T, batch_size, n*n)
        LQR_Q: (batch_size, n*n)
        LQR_R: (batch_size, m*m)
        x: (batch_size, T', n)
        return: (batch_size, T, n)
        """
        batch_size = B.size(0)
        B = B.view(batch_size, self.n, self.m)
        # calculate the inverse of LQR_R
        LQR_R_inv = torch.inverse(LQR_R)
        # calculate the optimal trajectory
        LQR_R_inv_B = torch.matmul(B, torch.matmul(LQR_R_inv, B.permute(0,2,1)))  # (batch_size, m*n)
        y = torch.matmul(G.view(self.T, batch_size, self.n, self.n), LQR_R_inv_B.unsqueeze(0).repeat(self.T, 1, 1, 1))  # (T, batch_size, n, n)
        if self.cfg['x_mode'] == 'All':
            y = torch.matmul(torch.matmul(LQR_Q.unsqueeze(0).repeat(self.T, 1, 1, 1), y), x.unsqueeze(3).transpose(1, 0)).squeeze(3)    # (T, batch_size, n)
        else:
            y = torch.matmul(torch.matmul(LQR_Q.unsqueeze(0).repeat(self.T, 1, 1, 1), y), x.unsqueeze(0).repeat(self.T, 1, 1, 1).transpose(-1,-2)).squeeze(3)    # (T, batch_size, n)
        y = self.input_proj(y)  # (T, batch_size, hidden_dim)
        y = self.blocks(y)  # (T, batch_size, hidden_dim)
        y = self.output_proj(y) # (T, batch_size, n)
        return y
        

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

        if config['AParam_model'] == 'Embedding':
            self.AParmaModel = A_ParamEmbedding(n, m, T, config)
        elif config['AParam_model'] == 'Formula':
            logger.error(f"AParam model {config['AParam_model']} not implemented")
            raise NotImplementedError(f"AParam model {config['AParam_model']} not implemented")
        elif config['AParam_model'] == 'MLP':
            self.AParmaModel = A_ParamMLP(n, T, config)

        input_dim = config['AParam_model_params'].get('output_size', n)
        self.GParamModel = GParamModel(n, m, T, input_dim, config, logger)
        self.FinalTrans = FinalTrans(n, m, T, config, logger)
        self.FNO = FNO2d(
            in_channels=1,
            out_channels=1,
            n_modes_height=16,
            n_modes_width=16,
            hidden_channels=64
        )


    def _get_batch(self, batch):
        x = batch['input_x']
        U = batch['input_u']
        A_true = batch['A']
        B_true = batch['B']
        alpha_true = batch['alphas']
        LQR_Q = batch['LQR_Q']
        LAR_R = batch['LQR_R']
        optimal_controls = batch['optimal_controls']
        return x, U, A_true, B_true, alpha_true, LQR_Q, LAR_R, optimal_controls
    
    def _get_regress_loss(self, A, B, alpha, A_true, B_true, alpha_true, config, logger):
        loss_type = config['loss_params']['regression_loss'].get('loss', 'MSE')
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

        reduction = config['loss_params']['regression_loss'].get('reduction', 'sum')
        if reduction == 'mean':
            return torch.mean(torch.stack([loss_A, loss_B, loss_alpha]))
        elif reduction == 'sum':
            return torch.sum(torch.stack([loss_A, loss_B, loss_alpha]))
        else:
            logger.error(f"Reduction type {reduction} not implemented")
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    

    def _get_label_loss(self, out, optimal_controls, config, logger):
        loss_type = config['loss_params']['label_loss'].get('loss', 'MSE')
        if loss_type == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
            # optimal_controls = optimal_controls.view(out.shape[0], -1)
        else:
            logger.error(f"Loss type {loss_type} not implemented")
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        loss = criterion(out, optimal_controls)
        return loss

    def _get_loss(self, regress_loss, label_loss, config, logger):
        reduction = config['loss_params'].get('reduction', 'sum')
        if reduction == 'mean':
            weight = config['loss_params']['label_loss'].get('weight', 0.5)
            return regress_loss*(1-weight) + label_loss*weight
        elif reduction == 'sum':
            return torch.sum(torch.stack([regress_loss, label_loss]))
        else:
            logger.error(f"Reduction type {reduction} not implemented")
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
        

    def forward(self, batch, logger):
        x, U, A_true, B_true, alpha_true, LQR_Q, LQR_R, optimal_controls = self._get_batch(batch)
        # print('x: ', x.shape)
        # print('LQR_Q: ', LQR_Q.shape)
        # print('LAR_R: ', LQR_R.shape)
        # print('optimal_controls: ', optimal_controls.shape)
        # A, B, alpha = self.param_regressor(x, U)
        A = A_true.view(-1, self.n*self.n)
        B = B_true.view(-1, self.n*self.m)
        alpha = alpha_true.view(-1, self.n)
        # A ,B and alpha loss
        regress_loss = self._get_regress_loss(A, B, alpha, A_true, B_true, alpha_true, self.cfg, logger)
        # print(f"regress_loss: {regress_loss}")
        A_emb = self.AParmaModel(U, A, alpha)  # (T, batch_size, n)
        # print(f"T_emb: {A_emb.shape}")
        G = self.GParamModel(A_emb) # (T, batch_size, n*n)
        # print(f"G: {G.shape}")
        
        skip_connection = self.cfg.get('skip_connection', False)
        if skip_connection:
            FNO_x = self.FinalTrans(B, G, LQR_Q, LQR_R, x) + U.transpose(0, 1)  # (T, batch_size, n)
        else:
            FNO_x = self.FinalTrans(B, G, LQR_Q, LQR_R, x)

        # out = FNO_x.transpose(0, 1)
        out = self.FNO(FNO_x.permute(1,0,2).unsqueeze(1)).squeeze(1)  # (batch_size, T, n)
        label_loss = self._get_label_loss(out, optimal_controls, self.cfg, logger)
        # print(f"label_loss: {label_loss}")
        return out, self._get_loss(regress_loss, label_loss, self.cfg, logger), regress_loss, label_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      