from typing import List, Optional

import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    table = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name not in table:
        raise ValueError(f"Neznáma aktivácia: {name}")
    return table[name]

class ConfigurableCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        activation: str = "relu",
        use_batch_norm: bool = False,
        pooling: str = "max",
        dropout: float = 0.0,
        fc_units: int = 128,
    ):
        super().__init__()

        if filters is None:
            filters = [32, 64]

        layers: List[nn.Module] = []
        current_channels = in_channels
        current_size = input_size
        padding = kernel_size // 2

        for num_filters in filters:
            layers.append(
                nn.Conv2d(
                    current_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(get_activation(activation))

            if current_size >= 2:
                if pooling == "max":
                    layers.append(nn.MaxPool2d(2, 2))
                elif pooling == "avg":
                    layers.append(nn.AvgPool2d(2, 2))
                else:
                    raise ValueError(f"Neznámy pooling: {pooling}")
                current_size = current_size // 2

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            current_channels = num_filters

        self.features = nn.Sequential(*layers)

        flat_size = current_channels * current_size * current_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_units),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> List[dict]:
        summary = []
        for i, layer in enumerate(self.features):
            summary.append(
                {
                    "index": i,
                    "type": layer.__class__.__name__,
                    "details": str(layer),
                }
            )
        for i, layer in enumerate(self.classifier):
            summary.append(
                {
                    "index": len(self.features) + i,
                    "type": layer.__class__.__name__,
                    "details": str(layer),
                }
            )
        return summary
