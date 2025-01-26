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

        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,  # input_shape (T, batch_size, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()

    def forward(self, x):
        """
        x: (T, batch_size, input_dim)
        return: (T, batch_size, output_dim)
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)  
        # output: (T, batch_size, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        # c_n: (num_layers, batch_size, hidden_dim)
        
        # => (T, batch_size, output_dim)
        y = self.dropout(self.act(self.fc(output)))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)
        
        return y
    
class RNNGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 dropout_rate=0.1, norm_type='None',activation='ReLU', config=None):
        """
        input_dim: input_dim of each time step
        hidden_dim: hidden_dim of RNN
        output_dim: output_dim (n*n)
        num_layers: number of layers of RNN
        """
        super(RNNGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,  # input_shape (T, batch_size, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()

    def forward(self, x):
        """
        x: (T, batch_size, input_dim)
        return: (T, batch_size, output_dim)
        """
        # RNN forward
        output, h_n = self.rnn(x)  
        # output: (T, batch_size, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        # c_n: (num_layers, batch_size, hidden_dim)
        
        # => (T, batch_size, output_dim)
        y = self.fc(output)
        y = self.dropout(self.act(y))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)
        
        return y
    
class GRUGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, 
                 dropout_rate=0.1, norm_type='None',activation='ReLU', config=None):
        """
        input_dim: input_dim of each time step
        hidden_dim: hidden_dim of GRU
        output_dim: output_dim (n*n)
        num_layers: number of layers of GRU
        """
        super(GRUGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,  # input_shape (T, batch_size, input_dim)
            bidirectional=config['bidirectional']
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = [nn.ReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid()][['ReLU', 'Tanh', 'GELU', 'Sigmoid'].index(activation)]
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_type = norm_type
        self.norm = [nn.BatchNorm1d(output_dim), nn.LayerNorm(output_dim)][['BatchNorm', 'LayerNorm'].index(norm_type)] if norm_type is not None else nn.Identity()
        
    def forward(self, x):
        """
        x: (T, batch_size, input_dim)
        return: (T, batch_size, output_dim)
        """
        # GRU forward
        output, h_n = self.gru(x)  
        # output: (T, batch_size, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        
        # => (T, batch_size, output_dim)
        y = self.fc(output)
        y = self.dropout(self.act(y))
        if self.norm_type == 'BatchNorm':
            y = self.norm(y.permute(1, 2, 0)).permute(2, 0, 1)
        elif self.norm_type == 'LayerNorm':
            y = self.norm(y)

        return y
    

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(5, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
