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