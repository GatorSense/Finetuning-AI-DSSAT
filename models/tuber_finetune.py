import torch
import torch.nn as nn
from models.tuber_model import TuberTransformer


class TuberTransformerWithExtraEncoders(nn.Module):
    """
    Fine-tuning wrapper for TuberTransformer:
      - loads pretrained TuberTransformer (base)
      - freezes base encoder
      - adds extra TransformerEncoder layers (trainable)
      - reuses base decoder + Softplus output activation

    This is a copy of utils/finetunemodel.py kept here so that all model
    definitions live under models/.  The canonical source in
    utils/finetunemodel.py is NOT removed (kept for backward-compat).
    """

    def __init__(
        self,
        pretrained_state_dict: dict,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        base_num_layers: int = 4,
        num_new_layers: int = 1,
        dropout: float = 0.2,
        seq_len: int = 160,
    ):
        super().__init__()

        # ---- Base model (must match training config) ----
        self.base = TuberTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=base_num_layers,
            dropout=dropout,
            seq_len=seq_len,
        )
        self.base.load_state_dict(pretrained_state_dict)

        # ---- Freeze base encoder ----
        for p in self.base.encoder.parameters():
            p.requires_grad = False

        # ---- Extra encoder layers (trainable) ----
        extra_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.extra_encoder = nn.TransformerEncoder(extra_layer, num_layers=num_new_layers)

        # Reuse base components
        self.input_proj = self.base.input_proj
        self.pos_enc    = self.base.pos_enc
        self.decoder    = self.base.decoder
        self.out_act    = self.base.out_act

        self.seq_len = seq_len

        # Buffers from base model (masks)
        self.causal_mask = self.base.causal_mask
        self.padding_mask_table = self.base.padding_mask_table

    def forward(self, x: torch.Tensor, xlens: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, T, F)
        xlens : (B,)
        returns: (B, T) predicted daily diffs (normalized scale if your base was trained that way)
        """
        B, T, _ = x.shape

        # Causal mask must be on same device
        src_mask = self.causal_mask[:T, :T].to(x.device)

        # padding mask table must be on same device as indices
        xl = torch.clamp(xlens, 0, T).long()
        padding_table = self.padding_mask_table.to(x.device)
        src_key_padding_mask = padding_table[xl][:, :T]  # (B,T) bool

        # Base encoder (frozen)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.base.encoder(
            h,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # New trainable layers
        h = self.extra_encoder(
            h,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Decoder + non-negativity
        y = self.decoder(h).squeeze(-1)
        return self.out_act(y)
