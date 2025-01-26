import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO2d, FNO1d
# from layer.layers import FNO1d
from layer.layers import ResidualMLPBlock
from layer.layers import LSTMGenerator, RNNGenerator, GRUGenerator
from utils.utils import LpLoss, UnitGaussianNormalizer
from layer.seq2seq import A2GEncoder, G2OutDecoder, A2G2OutSeq2Seq

class SEQParamRegressor(nn.Module):
    def __init__(self,
                n, m, T,
                config):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config
        model_type, hidden_size, num_layers, dropout_rate, norm_type, activation, bidirectional = self._get_config(config)
        self.bidirectional = bidirectional
        if config['use_U']:
            in_d = n + m
        else:
            in_d = n
        if model_type == 'RNN':
            self.seq = nn.RNN(
                input_size=in_d, 
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,         # (batch, seq_len, input_size)
                bidirectional=bidirectional
        )
        elif model_type == 'LSTM':
            self.seq = nn.LSTM(
                input_size=in_d,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif model_type == 'GRU':
            self.seq = nn.GRU(
                input_size=in_d,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        prev_dim = hidden_size*2 if bidirectional else hidden_size
        output_dim = self.n*self.n + self.n*self.m + self.n
        # self.fc = nn.Linear(prev_dim, output_dim)
        # self.act = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()][['ReLU', 'GELU',  'Tanh', 'Sigmoid'].index(activation)]
        # self.dropout = nn.Dropout(dropout_rate)
        # self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()
    
        blocks = []
        block_layers = config['identification_model_params']['block_layers']
        use_residual = config['identification_model_params']['use_residual']
        mlp_hidden_size = config['identification_model_params']['mlp_hidden_size']
        for i in range(len(block_layers)):
            block = ResidualMLPBlock(
                in_dim=prev_dim,
                hidden_dims=block_layers[i] * [mlp_hidden_size],
                num_layers=block_layers[i],
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                activation=activation,
                use_residual=use_residual  
            )
            blocks.append(block)
            prev_dim = mlp_hidden_size
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(prev_dim, output_dim)
 
    def forward(self, x, U):
        """
        x: (batch_size, T, n)
        U: (batch_size, T, m)
        return: (batch_size, n*n + n*m + n)
        """
        x = x.view(x.size(0), -1, self.n)
        U = U.view(U.size(0), -1, self.m)
        if self.cfg['use_U']:
            x = torch.cat([x, U], dim=2)  # (batch_size, T, n+m)
        if self.seq.__class__.__name__ == 'LSTM':
            x, (h, c) = self.seq(x) # x: (batch_size, T, hidden_size), h: (num_layers, batch_size, hidden_size)
        else:
            x, h = self.seq(x) # x: (batch_size, T, hidden_size), h: (num_layers, batch_size, hidden_size)
        if self.bidirectional:
            forward_hn = h[-2]
            backward_hn = h[-1]
            hn = torch.cat([forward_hn, backward_hn], dim=1)
        else:
            hn = h[-1]
        hn = self.blocks(hn)
        hn = self.fc(hn)
        # hn = self.act(self.fc(hn))
        # hn = self.dropout(hn)
        # hn = self.norm(hn)
    
        nn_ = self.n*self.n
        nm_ = self.n*self.m
        A_flat = hn[:, :nn_]
        B_flat = hn[:, nn_ : nn_ + nm_]
        alpha_flat = hn[:, nn_ + nm_:]
        
        return A_flat, B_flat, alpha_flat

    
    def _get_config(self, config):
        model_type = config['identification_model']
        hidden_size = config['identification_model_params'].get('hidden_size', 64)
        num_layers = config['identification_model_params'].get('num_layers', 2)
        dropout_rate = config['identification_model_params'].get('dropout_rate', 0.0)
        norm_type = config['identification_model_params'].get('norm_type', None)
        activation = config['identification_model_params'].get('activation', 'ReLU')
        bidirectional = config['identification_model_params'].get('bidirectional', False)
        return model_type, hidden_size, num_layers, dropout_rate, norm_type, activation, bidirectional
    


class MLPParamRegressor(nn.Module):
    def __init__(self,
                 n, m, T,
                 config
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        
        # in_dim = n + m*T   # T * dim + dim  (batch_size, T*dim + dim)
        in_dim = n*T
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
        inp = x_flat
        # U_flat = U.view(U.size(0), -1)
        # inp = torch.cat([x_flat, U_flat], dim=1)  # shape=(batch_size, in_dim), 

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
        hidden_size, dropout_rate, norm_type, activation, use_residual, use_x = self._get_config(config)
        self.output_size = config['AParam_model_params'].get('output_size', n)
        self.use_x = config['AParam_model_params'].get('use_x', False)
        blocks = []
        block_layers = config['AParam_model_params']['block_layers']
        prev_dim = hidden_size
        mlp_hidden_size = config['AParam_model_params']['mlp_hidden_size']
        if use_x:
            self.fc_x = nn.Linear(n, hidden_size//2)
            self.fc_alphas = nn.Linear(n, hidden_size//2)
            for i in range(len(block_layers)):
                block = ResidualMLPBlock(
                    in_dim=prev_dim,
                    hidden_dims=block_layers[i] * [mlp_hidden_size],
                    num_layers=block_layers[i],
                    dropout_rate=dropout_rate,
                    norm_type=norm_type,
                    activation=activation,
                    use_residual=use_residual  
                )
                prev_dim = mlp_hidden_size
                blocks.append(block)
            blocks.append(nn.Linear(prev_dim, self.output_size))
            self.act = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()][['ReLU', 'GELU',  'Tanh', 'Sigmoid'].index(activation)]
            blocks.append(self.act)
            blocks.append(nn.Dropout(dropout_rate))
            self.blocks = nn.Sequential(*blocks)
            self.norm_type = norm_type
            self.norm = [nn.BatchNorm1d(self.output_size), nn.LayerNorm(self.output_size)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()
        else:
            self.Embedding = nn.Embedding(T, hidden_size//2)
            self.fc1 = nn.Linear(n*2, hidden_size)
            self.fc2 = nn.Linear(n, hidden_size//2)
            for i in range(len(block_layers)):
                block = ResidualMLPBlock(
                    in_dim=prev_dim,
                    hidden_dims=block_layers[i] * [mlp_hidden_size],
                    num_layers=block_layers[i],
                    dropout_rate=dropout_rate,
                    norm_type=norm_type,
                    activation=activation,
                    use_residual=use_residual  
                )
                prev_dim = mlp_hidden_size
                blocks.append(block)

            blocks.append(nn.Linear(prev_dim, self.output_size))
            self.act = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()][['ReLU', 'GELU',  'Tanh', 'Sigmoid'].index(activation)]
            blocks.append(self.act)
            blocks.append(nn.Dropout(dropout_rate))
            self.blocks = nn.Sequential(*blocks)
            self.norm_type = norm_type
            self.norm = [nn.BatchNorm1d(self.output_size), nn.LayerNorm(self.output_size)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()
            

    def forward(self, x, A_flat, alpha):
        if self.use_x:
            emb0 = self.act(self.fc_x(x))  # (batch_size, T, hidden_size//2)
            emb1 = self.act(self.fc_alphas(alpha.unsqueeze(1).repeat(1, self.T, 1)))  # (batch_size, T, hidden_size//2)
            emb = torch.cat([emb0, emb1], dim=2)  # (batch_size, T, hidden_size)
            emb = self.blocks(emb).transpose(0, 1)  # (T, batch_size, output_size)
            if self.norm_type == 'BatchNorm':
                emb = self.norm(emb.permute(1, 2, 0)).permute(2, 0, 1)
            elif self.norm_type == 'LayerNorm':
                emb = self.norm(emb)
            return emb
        else:
            # calculate the eigenvalues of A
            A0 = A_flat.view(A_flat.shape[0], self.n, self.n) - torch.diag_embed(alpha)
            eigenvalues, eigenvectors = torch.linalg.eig(A0) # (batch_size, n), (batch_size, n, n)
            eigenvalues_real = torch.view_as_real(eigenvalues)  # (batch_size, n, 2)
            eigenvalues_real = eigenvalues_real.view(eigenvalues_real.shape[0], -1)  # (batch_size, n * 2)
            A0_emb = self.act(self.fc1(eigenvalues_real)) # (batch_size, n)
            alpha_emb = self.act(self.fc2(alpha.unsqueeze(1).repeat(1, self.T - 1, 1))) # (batch_size, T - 1, hidden_size//2)
            Ts = torch.arange(self.T-1).unsqueeze(0).repeat(A_flat.size(0), 1).to(A_flat.device)
            T_emb = self.Embedding(Ts) # (batch_size, T-1, hidden_size//2)
            T_emb = torch.cat([T_emb, alpha_emb], dim=2)  # (batch_size, T-1, hidden_size)
            T_emb = torch.cat([A0_emb.unsqueeze(1), T_emb], dim=1)  # (batch_size, T, hidden_size)
            emb = self.blocks(T_emb).transpose(0, 1)  # (T, batch_size, output_size)

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
        use_x = config['AParam_model_params'].get('use_x', False)
        return  hidden_size, dropout_rate, norm_type, activation, use_residual, use_x
        
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

        return x.transpose(0, 1)  # (T, batch_size, output_size)


class GParamModel(nn.Module):
    def __init__(self, n, m, T, input_dim, 
                 config, logger, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.cfg = config
        self.n = n
        self.m = m
        self.T = T
        model_type, hidden_size, num_layers, seq2seq, dropout_rate, norm_type, activation = self._get_config(config)
       
        if model_type == 'LSTM':
            self.seq_model = LSTMGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=config['GParam_model_params']['output_size'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'RNN':
            self.seq_model = RNNGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=config['GParam_model_params']['output_size'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'GRU':
            self.seq_model = GRUGenerator(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, output_dim=config['GParam_model_params']['output_size'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, config=config['GParam_model_params'])
        elif model_type == 'Transformer':
            self.seq_model = A2GEncoder(model_type=model_type, d_in=input_dim, d_model=hidden_size, num_layers=num_layers, config=config['GParam_model_params'], seq2seq=seq2seq)
        elif model_type == 'Formula':
            logger.error(f"Model type {model_type} not implemented")
            raise NotImplementedError(f"Model type {model_type} not implemented")
    
        # self.seq_model 
    def _get_config(self, config):
        model_type = config['GParam_model']
        hidden_size = config['GParam_model_params'].get('hidden_size', 64)
        num_layers = config['GParam_model_params'].get('num_layers', 2)
        seq2seq = config['seq2seq']
        dropout_rate = config['GParam_model_params'].get('dropout_rate', 0.0)
        norm_type = config['GParam_model_params'].get('norm_type', None)
        activation = config['GParam_model_params'].get('activation', 'ReLU')
        return model_type, hidden_size, num_layers, seq2seq, dropout_rate, norm_type, activation

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
        QRB_size, hidden_size, dropout_rate, norm_type, activation= self._get_config(config)
        self.fc_B = nn.Linear(n*m, hidden_size)
        self.fc_LQR_Q = nn.Linear(n*n, hidden_size)
        self.fc_LQR_R = nn.Linear(m*m, hidden_size)
        self.fc = nn.Linear(hidden_size*3, QRB_size)
        self.fc_x = ResidualMLPBlock(in_dim = n, hidden_dims=[config['FINAL_model_params']['hidden_size_X']]*config['FINAL_model_params']['num_layers_X'], num_layers=config['FINAL_model_params']['num_layers_X'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, use_residual=config['FINAL_model_params']['use_residual'])
        self.out_x = nn.Linear(config['FINAL_model_params']['hidden_size_X'], config['FINAL_model_params']['output_size_X'])
        self.fc_G = ResidualMLPBlock(in_dim = config['GParam_model_params']['output_size'], hidden_dims=[config['FINAL_model_params']['hidden_size_G']]*config['FINAL_model_params']['num_layers_G'], num_layers=config['FINAL_model_params']['num_layers_G'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, use_residual=config['FINAL_model_params']['use_residual'])
        self.out_G = nn.Linear(config['FINAL_model_params']['hidden_size_G'], config['FINAL_model_params']['output_size_G'])
        self.fc_U = ResidualMLPBlock(in_dim = m, hidden_dims=[config['FINAL_model_params']['hidden_size_U']]*config['FINAL_model_params']['num_layers_U'], num_layers=config['FINAL_model_params']['num_layers_U'], dropout_rate=dropout_rate, norm_type=norm_type, activation=activation, use_residual=config['FINAL_model_params']['use_residual'])
        self.out_U = nn.Linear(config['FINAL_model_params']['hidden_size_U'], config['FINAL_model_params']['output_size_U'])
        self.act = [nn.ReLU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()][['ReLU', 'GELU',  'Tanh', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = [nn.BatchNorm1d(QRB_size), nn.LayerNorm(QRB_size)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()
        blocks = []
        prev_dim = config['FINAL_model_params']['output_size_X']+ config['FINAL_model_params']['output_size_G']+ config['FINAL_model_params']['QRB_size']
        if config['use_U']:
            prev_dim += config['FINAL_model_params']['output_size_U']
        self.output_size = prev_dim
        use_residual = config['FINAL_model_params']['use_residual']
        mlp_hidden_size = config['FINAL_model_params']['hidden_size']
        block_layers = config['FINAL_model_params']['block_layers']
        if len(block_layers) > 0:
            blocks.append(nn.Linear(prev_dim, mlp_hidden_size))
            prev_dim = mlp_hidden_size
            self.output_size = config['FINAL_model_params']['output_size']
            for i in range(len(block_layers)):
                block = ResidualMLPBlock(
                    in_dim=prev_dim,
                    hidden_dims=block_layers[i] * [mlp_hidden_size],
                    num_layers=block_layers[i],
                    dropout_rate=dropout_rate,
                    norm_type=norm_type,
                    activation=activation,
                    use_residual=use_residual  
                )
                prev_dim = mlp_hidden_size
                blocks.append(block)
            blocks.append(nn.Linear(prev_dim, self.output_size))
        self.blocks = nn.Sequential(*blocks)



    def _get_config(self, config):
        hidden_size = config['FINAL_model_params'].get('hidden_size', 64)
        dropout_rate = config['FINAL_model_params'].get('dropout_rate', 0.0)
        norm_type = config['FINAL_model_params'].get('norm_type', None)
        activation = config['FINAL_model_params'].get('activation', 'ReLU')
        QRB_size = config['FINAL_model_params'].get('QRB_size', 3)
        return QRB_size, hidden_size, dropout_rate, norm_type, activation
        
    def forward(self, B, G, LQR_Q, LQR_R, x, U):
        """
        B: (batch_size, n*m)
        G: (T, batch_size, n)
        LQR_Q: (batch_size, n*n)
        LQR_R: (batch_size, m*m)
        x: (batch_size, T, n)
        return: (batch_size, T, n)
        """
        emb1 = self.act(self.fc_LQR_Q(LQR_Q.view(-1, self.n*self.n))) # q (batch_size, 25), (batch_size, 64) 
        emb2 = self.act(self.fc_LQR_R(LQR_R.view(-1, self.m*self.m))) # r (batch_size, 25), (batch_size, 64)
        emb3 = self.act(self.fc_B(B)) # B (batch_size, 25), (batch_size, 64)
        emb = torch.cat([emb1, emb2, emb3], dim=1).unsqueeze(0).repeat(self.T, 1, 1) # (T, batch_size, 128)
        emb = self.act(self.fc(emb)) # (T, batch_size, qrb_size)
        emb = self.dropout(emb)
        out = self.norm(emb) # (T, batch_size, 5)
        x = self.fc_x(x.permute(1,0,2)) # (T, batch_size, n)
        x = self.out_x(x)
        G = self.fc_G(G) # (T, batch_size, n)
        G = self.out_G(G)
        U = self.fc_U(U.permute(1,0,2)) # (T, batch_size, m)
        U = self.out_U(U)
        if self.cfg.get('use_U', False):
            out = torch.cat([out, x, G, U], dim=2)
        else:
            out = torch.cat([out, x, G], dim=2)
        out = self.blocks(out)
        return out
        

# Define the CFNO model 
class CFNO(nn.Module):
    def __init__(self,
                 n, m, T,
                 config, logger
                 ):
        super().__init__()
        self.training = True
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config
        if config['identification_model'] == 'MLP':
            self.param_regressor = MLPParamRegressor(n, m, T, config)
        elif (config['identification_model'] == 'RNN') or (config['identification_model'] == 'LSTM') or (config['identification_model'] == 'GRU'):
            self.param_regressor = SEQParamRegressor(n, m, T, config)
            
        if config['AParam_model'] == 'Embedding':
            self.AParmaModel = A_ParamEmbedding(n, m, T, config)
        elif config['AParam_model'] == 'Formula':
            logger.error(f"AParam model {config['AParam_model']} not implemented")
            raise NotImplementedError(f"AParam model {config['AParam_model']} not implemented")
        elif config['AParam_model'] == 'MLP':
            self.AParmaModel = A_ParamMLP(n, T, config)

        input_dim = config['AParam_model_params'].get('output_size', n)
        if config['seq2seq']:
            self.GParamModel = A2G2OutSeq2Seq(config=config['Seq2Seq_model_params'], d_in=input_dim, d_out=m, d_model=config['Seq2Seq_model_params']['hidden_size'], num_layers=config['Seq2Seq_model_params']['num_layers'], model_type=config['Seq2Seq_model'])
        else:
            self.GParamModel = GParamModel(n, m, T, input_dim, config, logger)
        self.FinalTrans = FinalTrans(n, m, T, config, logger)

        FNO_out_size = self.FinalTrans.output_size

        self.FNO = FNO1d(
            in_channels=FNO_out_size,
            out_channels=self.m,
            n_modes_height=16,
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
        elif loss_type == 'Lploss':
            criterion = LpLoss()
        else:
            logger.error(f"Loss type {loss_type} not implemented")
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        A_true = A_true.view(A.shape[0], -1)
        B_true = B_true.view(B.shape[0], -1)     
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
        elif loss_type == 'Lploss':
            criterion = LpLoss()
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
        

    def forward(self, batch, norms, logger):
        x, U, A_true, B_true, alpha_true, LQR_Q, LQR_R, optimal_controls = self._get_batch(batch)
        A, B, alpha = self.param_regressor(x, U)
        # A ,B and alpha loss
        regress_loss = self._get_regress_loss(A, B, alpha, A_true, B_true, alpha_true, self.cfg, logger)

        A_emb = self.AParmaModel(x, A, alpha)  # (T, batch_size, n)
        # print(f"T_emb: {A_emb.shape}")
        if self.cfg['seq2seq']:
            if self.training:
                G = self.GParamModel(A_emb, optimal_controls.permute(1,0,2))  # (T, batch_size, n)
            else:
                G = self.GParamModel.generate(A_emb, A_emb.size(0), d_out=self.n)  # (T, batch_size, n)
        else:
            G = self.GParamModel(A_emb) # (T, batch_size, n)
        FNO_x = self.FinalTrans(B, G, LQR_Q, LQR_R, x, U)  # (T, batch_size, n)
        FNO_x = nn.Tanh()(FNO_x)
        out = self.FNO(FNO_x.permute(1,2,0)).permute(0, 2, 1)  # (batch_size, T, n)
    
        if norms is not None:
            out = norms['optimal_controls'].decode(out)
            optimal_controls = norms['optimal_controls'].decode(optimal_controls)

        label_loss = self._get_label_loss(out, optimal_controls, self.cfg, logger)
        # print(f"label_loss: {label_loss}")
        return out, self._get_loss(regress_loss, label_loss, self.cfg, logger), regress_loss, label_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      





# Define the CFNO model 
class baseFNO(nn.Module):
    def __init__(self,
                 n, m, T,
                 config, logger
                 ):
        super().__init__()
        self.n = n
        self.m = m
        self.T = T
        self.cfg = config

        self.fc1 = nn.Linear(n*n, 64)
        self.fc2 = nn.Linear(m*m, 64)
        self.fc3 = nn.Linear(128, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(2)
        self.FNO = FNO1d(
            in_channels=2,
            out_channels=self.m,
            n_modes_height=16,
            hidden_channels=64
        )
        # self.FNO = FNO1d(modes=16, width=64)
      
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
        elif loss_type == 'Lploss':
            criterion = LpLoss()
        else:
            logger.error(f"Loss type {loss_type} not implemented")
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        A_true = A_true.view(A.shape[0], -1)
        B_true = B_true.view(B.shape[0], -1)        
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
        elif loss_type == 'Lploss':
            criterion = LpLoss()
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
        
    # FNO baseline
    def forward(self, batch, norms, logger):
        x, U, A_true, B_true, alpha_true, LQR_Q, LQR_R, optimal_controls = self._get_batch(batch)
        out = x
        # emb1 = self.act(self.fc1(LQR_Q.view(-1, self.n*self.n))) # q (batch_size, 25), (batch_size, 64) 
        # emb2 = self.act(self.fc2(LQR_R.view(-1, self.m*self.m))) # r (batch_size, 25), (batch_size, 64)
        # emb = torch.cat([emb1, emb2], dim=1).unsqueeze(1).repeat(1, self.T, 1) # (batch_size, T, 128)
        # emb = self.act(self.fc3(emb)) # (batch_size, T, 2)
        # emb = self.dropout(emb)
        # out = self.norm(emb) # (batch_size, T, 5)
        # out = torch.cat([out, x], dim=2)
  
        # out = self.FNO(out)
        out = self.FNO(out.permute(0, 2, 1)).permute(0, 2, 1) # (batch_size, T, n)   
        if norms is not None:
            out = norms['optimal_controls'].decode(out)
            optimal_controls = norms['optimal_controls'].decode(optimal_controls)
        label_loss = self._get_label_loss(out, optimal_controls, self.cfg, logger)
        return out, label_loss, label_loss, label_loss
