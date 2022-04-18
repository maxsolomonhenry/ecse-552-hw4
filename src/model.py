import pytorch_lightning as pl
import torch.nn as nn
import torch

class BaselineMlp(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = self.init_encoder(n_input, n_hidden, n_output)

    def init_encoder(self, n_input, n_hidden, n_output):
        # Build simple neural net with ReLU and dropout. 
        #
        # `n_hidden` should be an array with (int) n_neurons per layer.
        n_layers = len(n_hidden)

        # First layer to accept data.
        all_layers = [nn.Linear(n_input, n_hidden[0]), nn.ReLU()]

        # Build all hidden layers, applying ReLU and dropout.
        for idx in range(n_layers - 1):
            n_in = n_hidden[idx]
            n_out = n_hidden[idx + 1]
            all_layers.extend([nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout()])

        # No dropout, activation for final layer.
        all_layers.extend([nn.Linear(n_hidden[-1], n_output)])

        return nn.Sequential(*all_layers)

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters, le=1e-3)