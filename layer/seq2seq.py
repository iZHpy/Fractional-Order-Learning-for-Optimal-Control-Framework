import torch
import torch.nn as nn
import math

# 1) A->G Encoder (eg: TransformerEncoder or Any Encoder)
class A2GEncoder(nn.Module):
    def __init__(self, model_type='Transformer', d_in=32, d_model=64, num_layers=2, config=None, seq2seq=False):
        super().__init__()
        self.model_type = model_type
        layers = []
        if model_type == 'Transformer':
            layers.append(nn.Linear(d_in, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config['nhead'])
            layers.append(nn.TransformerEncoder(encoder_layer, num_layers=num_layers))
            if seq2seq == False:
                layers.append(nn.Linear(d_model, config['output_size']))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config['dropout_rate']))
                layers.append(nn.LayerNorm(config['output_size'])) 
        elif model_type == 'RNN':
            layers.append(nn.RNN(d_in, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        elif model_type == 'LSTM':
            layers.append(nn.LSTM(d_in, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        elif model_type == 'GRU':
            layers.append(nn.GRU(d_in, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, A):
        """
        A: [src_seq_len, batch_size, d_in]
        return G: [src_seq_len, batch_size, d_model] if seq2seq=True else [src_seq_len, batch_size, output_size]
        """
        if (self.model_type == 'RNN') or (self.model_type == 'LSTM') or (self.model_type == 'GRU'):
            G, _ = self.encoder(A)
        else:
            G = self.encoder(A)   # treat as "encoder output"
        return G

# 2) G->out Decoder (示例: TransformerDecoder + Any Decoder)
class G2OutDecoder(nn.Module):
    def __init__(self, model_type='Transformer', d_model=64, d_out=5, num_layers=2, config=None):
        super().__init__()
        self.d_model = d_model
        layers = []
        # 如果目标 out 是离散 token，这里可以 Embedding
        # 也可以直接把"tgt序列"以embedding形式输入
        if model_type == 'RNN':
            layers.append(nn.RNN(d_model, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        elif model_type == 'LSTM':
            layers.append(nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        elif model_type == 'GRU':
            layers.append(nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=False, bidirectional=config['bidirectional']))
        elif model_type == 'Transformer':
            self.decoder_input_proj = nn.Linear(d_out, d_model)
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=config['nhead'])
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 最后映射到 vocab_size (若是字符/词预测)
        self.fc_out = nn.Linear(d_model, d_out)
    
    def forward(self, G_memory, tgt_input, tgt_mask=None):
        """
        G_memory: 来自 A2GTransformer 的输出, shape: [src_seq_len, batch_size, d_model]
        tgt_input: decoder 的输入 (token IDs), shape: [tgt_seq_len, batch_size]
        tgt_mask:  (可选) 用来做 causal mask, [tgt_seq_len, tgt_seq_len]
        
        return: logits, [tgt_seq_len, batch_size, vocab_size]
        """
        # (1) 目标序列 embedding
        tgt_emb = self.decoder_input_proj(tgt_input)  # [tgt_seq_len, batch_size, d_model]
        
        # (2) Decoder 前向
        #     memory=G_memory, tgt=tgt_emb
        out = self.decoder(tgt=tgt_emb, memory=G_memory, tgt_mask=tgt_mask)
        
        # (3) 映射到 out_dim
        y_pred = self.fc_out(out)
        return y_pred

# 3) 整体 A->G->out seq2seq
class A2G2OutSeq2Seq(nn.Module):
    def __init__(self, 
                 model_type=None,
                 d_in=32, 
                 d_out=5,
                 d_model=64, 
                 num_layers=2,
                 config=None):
        super().__init__()
        self.a2g = A2GEncoder(model_type=model_type,d_in=d_in, d_model=d_model,num_layers=num_layers,config=config,seq2seq=True)
        self.g2out = G2OutDecoder(model_type=model_type, d_out=d_out, d_model=d_model, num_layers=num_layers, config=config)
        
    def forward(self, A, tgt_input, tgt_mask=None):
        """
        A:         [src_seq_len, batch_size, d_in]
        tgt_input: [tgt_seq_len, batch_size] (目标序列 ID)
        tgt_mask:  (可选) [tgt_seq_len, tgt_seq_len], 用于因果mask
        
        return: logits => [tgt_seq_len, batch_size, tgt_vocab_size]
        """
        G = self.a2g(A)  # [src_seq_len, batch_size, d_model]
        shifted_tgt_input = torch.roll(tgt_input, shifts=1, dims=0)
        shifted_tgt_input[0] = 0
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(tgt_input.device) 
        logits = self.g2out(G, shifted_tgt_input, tgt_mask=tgt_mask)
        return logits

    def generate(self, A, max_len=10, d_out=5):
        """
        A: [src_seq_len, batch_size, d_in]
        max_len: 生成序列的最大长度
        
        return: y_pred => [tgt_seq_len, batch_size]
        """
        G = self.a2g(A)
        tgt_input = torch.zeros(1, A.size(1), d_out).float().to(A.device)
        for i in range(max_len):
            logits = self.g2out(G, tgt_input)
            tgt_input = torch.cat([tgt_input, logits[-1].unsqueeze(0)], dim=0)
        return tgt_input[1:]

