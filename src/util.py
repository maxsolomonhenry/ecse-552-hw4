import matplotlib
import matplotlib.pyplot as plt

def plot_logger_metrics(logger):

    _, axs = plt.subplots(1, 1, figsize=(15, 5))
    font = {'size': 14}
    matplotlib.rc('font', **font)

    axs.plot(logger.metrics['train_loss'], lw=3, ms=8, marker='o', color='orange', label='Train')
    axs.set_title("Train/Val Loss")
    axs.set_ylabel("Loss")
    axs.plot(logger.metrics['val_loss'], lw=3, ms=10, marker='^', color='purple', label='Validation')
    axs.set_title('Train/Val Loss Over Time')
    axs.set_xlabel("Epochs")
    axs.grid()

    plt.legend(loc='lower right')
    plt.show()