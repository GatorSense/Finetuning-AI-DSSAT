import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for batch_first tensors:
      x shape: (B, T, d_model)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)               # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                         # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                         # odd dims

        # store as (1, max_len, d_model) so it broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TuberTransformer(nn.Module):
    """
    Upgraded tuber model:
      - batch_first=True (no permute)
      - causal mask precomputed
      - per-sample padding mask using xlens
      - bigger FFN (4*d_model)
      - non-negativity enforced via Softplus (smoother than ReLU)

    This is a copy of utils/model.py:TuberTransformer kept here so that
    all model definitions live under models/.  The canonical source in
    utils/model.py is NOT removed (kept for backward-compat with existing
    pickle files and training scripts).
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.2, seq_len: int = 160):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoder (two-layer MLP) + Softplus to enforce >=0
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.out_act = nn.Softplus(beta=1.0)  # smoother non-negativity than ReLU

        # Precompute masks once
        self.register_buffer("causal_mask", self._make_causal_mask(seq_len), persistent=False)
        self.register_buffer("padding_mask_table", self._make_padding_mask_table(seq_len), persistent=False)

    @staticmethod
    def _make_causal_mask(T: int) -> torch.Tensor:
        # bool mask: True means "block attention"
        return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _make_padding_mask_table(T: int) -> torch.Tensor:
        # padding_mask_table[L] returns a (T,) bool mask
        # where True means "this position is padding"
        table = torch.zeros(T + 1, T, dtype=torch.bool)
        for L in range(T + 1):
            table[L, L:] = True
        return table

    def forward(self, x: torch.Tensor, xlens: torch.Tensor,
                src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:     (B, T, F)
        xlens: (B,) actual lengths (<= T)
        returns: (B, T) daily tuber growth predictions
        """
        if xlens is None:
            raise ValueError("xlens must be provided for padding masking.")

        B, T, _ = x.shape
        if T > self.seq_len:
            raise ValueError(f"Input seq_len {T} exceeds model seq_len {self.seq_len}")

        if src_mask is None:
            src_mask = self.causal_mask[:T, :T]

        if src_key_padding_mask is None:
            # xlens may contain values 0..T, clamp for safety
            xl = torch.clamp(xlens, 0, T).to(torch.long)
            src_key_padding_mask = self.padding_mask_table[xl][:, :T]  # (B, T)

        x = self.input_proj(x)                      # (B, T, d_model)
        x = self.pos_enc(x)                         # (B, T, d_model)
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        y = self.decoder(x).squeeze(-1)             # (B, T)
        y = self.out_act(y)                         # enforce >= 0 smoothly
        return y
