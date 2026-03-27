import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_nf,
        out_dim,
        n_layers,
        n_heads,
        dropout,
        activation,
        dim_feedforward=128,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_dim, hidden_nf)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_nf,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Expects [Batch, Seq_Len, Features]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), nn.SiLU(), nn.Linear(hidden_nf, out_dim)
        )

    def forward(self, x, batch_index=None):
        """
        x: [Total_Nodes, 3] (PyG style) or [Batch, N, 3]
        If PyG style, we need to pad/reshape to [Batch, Max_N, 3]
        """
        # For simplicity, assuming x is [Batch, N, 3]
        # If input is [N, 3], add batch dim: x = x.unsqueeze(0)

        # Linear projection
        h = self.embedding(x)

        # Global self-attention
        # All nodes attend to all nodes (fully connected logic)
        h = self.transformer_encoder(h)

        # Final projection
        out = self.decoder(h)
        return out
