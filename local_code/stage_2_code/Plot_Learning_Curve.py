'''
Plots the learning convergence curves (training loss and accuracy per epoch).
'''

import matplotlib
matplotlib.use('Agg')  # save to file without opening a window
import matplotlib.pyplot as plt


def plot_learning_curve(history, save_path, title='MLP learning curve'):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history['train_loss'], marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title('Training loss')
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training accuracy')
    ax2.set_title('Training accuracy')
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('learning curve saved to', save_path)


def plot_loss_comparison(histories, save_path, title='Training loss by configuration'):
    '''histories: dict of {config name: history}'''
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, h in histories.items():
        ax.plot(range(1, len(h['train_loss']) + 1), h['train_loss'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('comparison plot saved to', save_path)