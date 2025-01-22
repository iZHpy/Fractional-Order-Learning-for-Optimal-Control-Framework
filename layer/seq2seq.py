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
        return G: [src_seq_len, batch_size, d_model]
        """
        if (self.model_type == 'RNN') or (self.model_type == 'LSTM') or (self.model_type == 'GRU'):
            G, _ = self.encoder(A)
        else:
            G = self.encoder(A)   # treat as "encoder output"
        return G

# 2) G->out Decoder (示例: TransformerDecoder + Any Decoder)
class G2OutDecoder(nn.Module):
    def __init__(self, d_model=64, vocab_size=50, nhead=2, num_layers=2, config=None):
        super().__init__()
        self.d_model = d_model
        
        # 如果目标 out 是离散 token，这里可以 Embedding
        # 也可以直接把"tgt序列"以embedding形式输入
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 最后映射到 vocab_size (若是字符/词预测)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, G_memory, tgt_input, tgt_mask=None):
        """
        G_memory: 来自 A2GTransformer 的输出, shape: [src_seq_len, batch_size, d_model]
        tgt_input: decoder 的输入 (token IDs), shape: [tgt_seq_len, batch_size]
        tgt_mask:  (可选) 用来做 causal mask, [tgt_seq_len, tgt_seq_len]
        
        return: logits, [tgt_seq_len, batch_size, vocab_size]
        """
        # (1) 目标序列 embedding
        tgt_emb = self.tgt_embedding(tgt_input)  # [tgt_seq_len, batch_size, d_model]
        
        # (2) Decoder 前向
        #     memory=G_memory, tgt=tgt_emb
        out = self.decoder(tgt=tgt_emb, memory=G_memory, tgt_mask=tgt_mask)
        
        # (3) 映射到 vocab_size
        logits = self.fc_out(out)
        return logits

# 3) 整体 A->G->out seq2seq
class A2G2OutSeq2Seq(nn.Module):
    def __init__(self, 
                 d_in=32, 
                 d_model=64, 
                 src_vocab_size=None,
                 tgt_vocab_size=50):
        super().__init__()
        self.a2g = A2GEncoder(d_in=d_in, d_model=d_model)
        self.g2out = G2OutDecoder(d_model=d_model, vocab_size=tgt_vocab_size)
    
    def forward(self, A, tgt_input, tgt_mask=None):
        """
        A:         [src_seq_len, batch_size, d_in]
        tgt_input: [tgt_seq_len, batch_size] (目标序列 ID)
        tgt_mask:  (可选) [tgt_seq_len, tgt_seq_len], 用于因果mask
        
        return: logits => [tgt_seq_len, batch_size, tgt_vocab_size]
        """
        G = self.a2g(A)  # [src_seq_len, batch_size, d_model]
        logits = self.g2out(G, tgt_input, tgt_mask=tgt_mask)
        return logits

