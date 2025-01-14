import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    """
    A block that supports multiple layers of Linear + Activation + Dropout + (optional)Norm,
    with residual connection if input and output dimensions are the same.
    """
    def __init__(self,
                 in_dim,
                 hidden_dims=None,
                 num_layers=2,
                 dropout_rate=0.0,
                 norm_type=None,    # None / 'batchnorm' / 'layernorm'
                 activation='relu',
                 use_residual=False):
        """
        params:
            in_dim      : input (and output) dimension, must be the same if use_residual
            hidden_dim  : dimension of hidden layers, can be the same as in_dim or different; if None, defaults to in_dim
            num_layers  : how many layers of Linear to stack
            dropout_rate: dropout probability
            norm_type   : 'batchnorm', 'layernorm' or None
            activation  : 'relu', 'gelu', etc.
            use_residual: whether to use residual connection; only possible if in_dim is the same as out_dim
        """
   
        super().__init__()
        if hidden_dims is None:
            self.hidden_dims = [in_dim] * num_layers
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.use_residual = use_residual

        out_dim = hidden_dims[-1] if hidden_dims is not None else in_dim

        # Build the MLP
        layers = []
        prev_dim = in_dim
        for i in range(num_layers):
            current_out_dim = hidden_dims[i]
            
            # 1) Linear
            linear_layer = nn.Linear(prev_dim, current_out_dim)
            layers.append(linear_layer)

            # 2) Activation
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'GELU':
                layers.append(nn.GELU())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            elif activation == 'Sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # 3) Dropout
            if dropout_rate > 1e-7:
                layers.append(nn.Dropout(dropout_rate))
            
            # 4) Optional normalization
            if norm_type == 'BatchNorm':
                # To apply BatchNorm1d, the input shape should be (batch, feature_dim)
                layers.append(nn.BatchNorm1d(current_out_dim))
            elif norm_type == 'LayerNorm':
                layers.append(nn.LayerNorm(current_out_dim))
            
            prev_dim = current_out_dim

        self.mlp = nn.Sequential(*layers)

        # if in_dim != out_dim: we cannot use residual connection even if use_residual=True
        # so we simply disable it
        if in_dim != out_dim:
            self.use_residual = False

    def forward(self, x):
        # x shape: (batch_size, in_dim)
        out = self.mlp(x)  # shape: (batch_size, out_dim) with out_dim == in_dim

        # if use_residual and in_dim == out_dim, add the input to the output
        if self.use_residual:
            out = x + out

        return out
    

class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 dropout_rate=0.1, norm_type='None',activation='ReLU', config=None):
        """
        input_dim: input_dim of each time step
        hidden_dim: hidden_dim of LSTM
        output_dim: output_dim (n*n)
        num_layers: number of layers of LSTM
        """
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq2sqe = config['seq2seq']
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input_shape (batch_size, T, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)]

    def forward(self, x):
        """
        x: (batch_size, T, input_dim)
        return: (batch_size, T, output_dim)
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)  
        # output: (batch_size, T, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        # c_n: (num_layers, batch_size, hidden_dim)
        
        # => (batch_size, T, output_dim)
        y = self.dropout(self.act(self.fc(output)))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.transpose(1,2)).transpose(1,2)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)
        
        return y
    
class RNNGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 dropout_rate=0.1, norm_type='None',activation='ReLU', config=None):
        """
        input_dim: input_dim of each time step
        hidden_dim: hidden_dim of LSTM
        output_dim: output_dim (n*n)
        num_layers: number of layers of LSTM
        """
        super(RNNGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq2sqe = config['seq2seq']
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input_shape (batch_size, T, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)]
        
    def forward(self, x):
        """
        x: (batch_size, T, input_dim)
        return: (batch_size, T, output_dim)
        """
        # LSTM forward
        output, h_n = self.rnn(x)  
        # output: (batch_size, T, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        
        # => (batch_size, T, output_dim)
        y = self.fc(output)
        y = self.dropout(self.act(y))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.transpose(1,2)).transpose(1,2)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)
        
        return y
    
class GRUGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 dropout_rate=0.1, norm_type='None',activation='ReLU', config=None):
        """
        input_dim: input_dim of each time step
        hidden_dim: hidden_dim of LSTM
        output_dim: output_dim (n*n)
        num_layers: number of layers of LSTM
        """
        super(GRUGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq2sqe = config['seq2seq']
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input_shape (batch_size, T, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)]
        
    def forward(self, x):
        """
        x: (batch_size, T, input_dim)
        return: (batch_size, T, output_dim)
        """
        # LSTM forward
        output, h_n = self.gru(x)  
        # output: (batch_size, T, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        
        # => (batch_size, T, output_dim)
        y = self.fc(output)
        y = self.dropout(self.act(y))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.transpose(1,2)).transpose(1,2)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)

        return y
    


class AtoGTransformer(nn.Module):
    """
    input: (src_seq_len, batch_size, d_model)
    output: (tgt_seq_len, batch_size, d_model)
    """
    def __init__(self, d_model, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()

        self.d_model = d_model
        # use nn.Linear to project the input to d_model
        self.src_linear = nn.Linear(d_model, d_model)
        self.tgt_linear = nn.Linear(d_model, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=False # input shape: (seq_len, batch_size, d_model)
        )

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, src, tgt):
        """
        src: [batch_size, src_seq_len, d_model]
        tgt: [batch_size, tgt_seq_len, d_model]
        returns: [batch_size, tgt_seq_len, d_model]
        """
        #  (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)  # -> (src_seq_len, batch_size, d_model)
        tgt = tgt.transpose(0, 1)  # -> (tgt_seq_len, batch_size, d_model)

        # project to d_model
        src_embed = self.src_linear(src)  # (src_seq_len, batch_size, d_model)
        tgt_embed = self.tgt_linear(tgt)

        # transformer forward
        out = self.transformer(
            src_embed, tgt_embed
            # src_key_padding_mask=None,
            # tgt_key_padding_mask=None,
            # memory_key_padding_mask=None
        )
        # out shape: (tgt_seq_len, batch_size, d_model)
        out = self.output_linear(out)

        # back to (batch_size, tgt_seq_len, d_model)
        out = out.transpose(0, 1)
        return out
