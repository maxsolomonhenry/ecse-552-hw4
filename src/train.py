from dict_logger import DictLogger
from data import get_dataloaders
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from util import plot_logger_metrics


def init_trainer(logger, max_epochs, do_early_stopping):
    
    print("Initializing trainer...")

    gpus = None
    auto_select_gpus = False

    is_colab = 'COLAB_GPU' in os.environ
    if is_colab:
        print('Colab detected. Using gpus...')
        gpus = -1
        auto_select_gpus = True

    callbacks = EarlyStopping('val_loss') if do_early_stopping else []

    trainer = pl.Trainer(gpus=gpus, auto_select_gpus=auto_select_gpus, 
                         callbacks=callbacks, logger=logger, 
                         max_epochs=max_epochs)

    return trainer


def train_model(model, max_epochs, n_past=8, batch_size=128, percent_train=0.8, 
                do_early_stopping=True):

    train_loader, val_loader = get_dataloaders(
        n_past=n_past, batch_size=batch_size, percent_train=percent_train
    )

    logger = DictLogger()
    trainer = init_trainer(logger, max_epochs, do_early_stopping)

    print('Training...')
    trainer.fit(model, train_loader, val_loader)
    print('Done!')

    plot_logger_metrics(logger)


if __name__ == "__main__":
    from data import get_dataloaders
    from model import BaselineMlp

    # Parameters.
    max_epochs = 100
    n_past = 8
    batch_size = 128
    n_hidden = [128, 64, 32, 16]

    # Load one batch to get the correct model dimensions.
    train_loader, val_loader = get_dataloaders(n_past=n_past, batch_size=1)
    train_iter = iter(train_loader)
    x, y = next(train_iter)

    n_input = x.flatten(1).shape[1]
    n_output = y.shape[1]

    model = BaselineMlp(
        n_input=n_input, n_hidden=n_hidden, n_output=n_output
    )

    train_model(model, max_epochs, n_past, batch_size, do_early_stopping=False)