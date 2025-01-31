import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO2d, FNO1d
from utils.utils import LpLoss


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
