import torch.nn as nn
from models.models  import TransformerModel

class SoilnTransformerWithExtraEncoders(nn.Module):
    def __init__(
        self,
        pretrained_state_dict,
        input_dim,
        d_model,
        nhead,
        base_num_layers,
        num_new_layers,
        dropout,
        seq_len,
    ):
        super().__init__()

        self.base = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=base_num_layers,
            dropout=dropout,
            seq_len=seq_len,
        )
        self.base.load_state_dict(pretrained_state_dict)

        for p in self.base.transformer_encoder.parameters():
            p.requires_grad = False

        extra_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.extra_encoder = nn.TransformerEncoder(
            extra_layer, num_layers=num_new_layers
        )

        self.input_encoding = self.base.input_encoding
        self.pos_encoder    = self.base.pos_encoder
        self.output_layer   = self.base.output_layer
        self.causal_mask    = self.base.causal_mask
        self.padding_mask   = self.base.padding_mask

    def forward(self, x, xlens):
        T = x.size(1)
        src_mask = self.causal_mask[:T, :T].to(x.device)
        src_key_padding_mask = self.padding_mask.to(x.device)[xlens][:, :T]

        h = self.input_encoding(x)
        h = self.pos_encoder(h)
        h = self.base.transformer_encoder(h, mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)
        h = self.extra_encoder(h, mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)

        return self.output_layer(h).squeeze(-1)
