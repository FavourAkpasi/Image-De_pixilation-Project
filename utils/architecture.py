# -*- coding: utf-8 -*-
"""
Author -- Favour Akpasi
Architectures file of Depixilation project.
"""

import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()

        cnn = []
        padding = (kernel_size - 1) // 2  # Add padding to maintain spatial dimensions
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=padding
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding
        )
        self.double()

    def forward(self, x):
        x = x.double()
        cnn_out = self.hidden_layers(x)
        predictions = self.output_layer(cnn_out)
        return predictions
