import math

import torch
import torch.nn as nn


class ModelMetrics:
    def __init__(self):
        self.train = self.Metrics()
        self.val = self.Metrics()

    class Metrics:
        def __init__(self):
            self.mse_loss = []
            self.mae_loss = []
            self.r2_score = []


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos = torch.zeros(max_len, d_model)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        pos = pos.unsqueeze(0)
        self.register_buffer("pe", pos)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, embedding_dim(d_model)]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


########################################################################################################


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, seq_len):
        super(TransformerModel, self).__init__()
        self.seq_len = seq_len
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=self.seq_len)

        # Initializing objects
        self.input_encoding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, 
            dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        
        # Test only if Linear does not suffice
        # self.output_layer = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 1)
        # )

        # Create causal and pad mask once during initialization
        self.register_buffer("causal_mask", self.generate_causal_mask())
        self.register_buffer("padding_mask", self.generate_padding_masks())

    def generate_causal_mask(self):
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool), diagonal=1)
        return mask

    def generate_padding_masks(self):
        pad_masks = torch.zeros(self.seq_len + 1, self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len + 1):
            pad_masks[i, i:] = True
        return pad_masks

    def forward(
        self,
        x,
        xlens=None,
        src_mask=None,
        src_key_padding_mask=None,
        return_hidden=False,   # <-- ADD THIS
    ):
        if xlens is None:
            raise AssertionError("xlens must be provided")

        if src_mask is None:
            src_mask = self.causal_mask

        if src_key_padding_mask is None:
            src_key_padding_mask = self.padding_mask[xlens]

        # ----- Input projection + positional encoding -----
        x = self.input_encoding(x)      # (B, S, d_model)
        x = self.pos_encoder(x)         # (B, S, d_model)

        # ----- Transformer encoder -----
        hidden = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )                               # (B, S, d_model)

        # ----- Original output head -----
        output = self.output_layer(hidden).squeeze(-1)  # (B, S)

        if return_hidden:
            return output, hidden       # <-- THIS IS THE KEY CHANGE

        return output
