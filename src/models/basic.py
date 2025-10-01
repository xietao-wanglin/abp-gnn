import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activation=nn.SiLU(),
        dropout=0.0,
        out_activation=False,
    ):
        super().__init__()

        layers = []
        dim = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(activation)
            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))
        if out_activation:
            layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
