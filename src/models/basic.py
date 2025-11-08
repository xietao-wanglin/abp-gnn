import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


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
        norm=True,
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

        if norm:
            layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConditionalMAF(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim=128, n_flows=4):
        super().__init__()

        transforms = []
        for _ in range(n_flows):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=2,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=nn.SiLU(),
                )
            )

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([input_dim]),
        )

    def forward(self, context, y):
        log_prob = self.flow.log_prob(inputs=y, context=context)
        return -log_prob

    def sample(self, context, n_samples=1):
        return self.flow.sample(n_samples, context)

    def sample_mean(self, context, n_samples=20):
        samples = self.flow.sample(n_samples, context)
        return samples.mean(dim=1)
