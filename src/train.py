if __name__ == "__main__":
    from data import get_dataloaders
    from model import BaselineMlp

    train_loader, val_loader = get_dataloaders(
        n_past=8, batch_size=128, percent_train=0.8
    )

    train_iter = iter(train_loader)
    x, y = next(train_iter)

    n_input = x.shape[1]
    n_output = y.shape[1]

    n_hidden = [256, 128, 64, 32]

    model = BaselineMlp(
        n_input=n_input, n_hidden=n_hidden, n_output=n_output
    )

    print(model)