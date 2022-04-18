import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import torch

class BaselineMlp(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = self.init_encoder(n_input, n_hidden, n_output)

        self.init_log()

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

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Flatten time series data.
        x = x.flatten(1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        self.train_accuracy(y_hat, y)
        self.log('train_mse', self.train_accuracy, on_step=False, 
                 on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Flatten time series data.
        x = x.flatten(1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        self.val_accuracy(y_hat, y)
        self.log('val_mse', self.val_accuracy, on_step=False, 
                 on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Flatten time series data.
        x = x.flatten(1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        self.test_accuracy(y_hat, y)
        self.log('test_mse', self.test_accuracy, on_step=False, 
                 on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def init_log(self):
        self.train_accuracy = torchmetrics.MeanSquaredError()
        self.val_accuracy = torchmetrics.MeanSquaredError()
        self.test_accuracy = torchmetrics.MeanSquaredError()