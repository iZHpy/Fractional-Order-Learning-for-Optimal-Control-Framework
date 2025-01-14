import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, encoder):
        """
        input_dim: Encoder input size of each time step
        hidden_dim: LSTM hidden size
        num_layers: LSTM layers
        """
        super(Encoder, self).__init__()
        
        # 一个多层 LSTM
        self.encoder = encoder
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        return:
          output: (batch_size, seq_len, hidden_dim)  
          (h_n, c_n): shape=(num_layers, batch_size, hidden_dim)
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, decoder):
        """
        input_dim: Decoder input size of each time step
        hidden_dim: LSTM hidden size
        num_layers: LSTM layers
        """
        super(Decoder, self).__init__()
        
        # 一个多层 LSTM
        self.decoder = decoder
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        return:
          output: (batch_size, seq_len, hidden_dim)  
          (h_n, c_n): shape=(num_layers, batch_size, hidden_dim)
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder,
                 teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
    
    def forward(self, src, tgt, max_len=None):
        """
        src: (batch_size, src_len, input_dim)   # Encoder 输入序列
        tgt: (batch_size, tgt_len, output_dim)  # Decoder 目标序列(训练时用), 其中 tgt_len 通常=目标序列长度
        max_len: 若推理时不知道目标长度, 可以设定一个最大长度.
        
        返回:
          outputs: (batch_size, tgt_len, output_dim)
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        output_dim = self.decoder.fc.out_features  # decoder 输出维度
        
        # --- 1) Encoder 前向 ---
        # 不需要 encoder_out 时, 也可以只接收 (h_n,c_n)
        _, (h_enc, c_enc) = self.encoder(src)
        
        # --- 2) 准备decoder输入与隐藏状态初值 ---
        # Decoder 初始隐藏状态 = Encoder 最后时刻隐状态
        dec_hidden = (h_enc, c_enc)
        
        # Decoder 初始输入 typically 是 <SOS> token 对于文本翻译
        # 这里简化: 用 tgt[:, 0, :] 或 全0向量 作为 "y_0"
        # shape: (batch_size, 1, output_dim)
        y_prev = tgt[:, 0, :].unsqueeze(1)
        
        # 用于存放输出
        outputs = torch.zeros(batch_size, tgt_len, output_dim, device=src.device)
        
        # --- 3) Decoder 逐时刻输出 ---
        for t in range(1, tgt_len):
            # Decoder 前向
            y_t, dec_hidden = self.decoder(y_prev, dec_hidden)
            # y_t: (batch_size, 1, output_dim)
            
            outputs[:, t, :] = y_t.squeeze(1)  # 存储第 t 步输出
            
            # -- Teacher forcing 判定 --
            use_teacher_forcing = (torch.rand(1).item() < self.teacher_forcing_ratio)
            if use_teacher_forcing:
                # 用真实标签 tgt[:, t, :] 作为下一时刻输入
                y_prev = tgt[:, t, :].unsqueeze(1)
            else:
                # 用模型预测的 y_t 作为下一时刻输入 (自回归)
                y_prev = y_t
        
        return outputs
