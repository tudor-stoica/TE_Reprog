import os
import matplotlib.pyplot as plt

def plot_metrics(epochs, acc_list, loss_list, out_dir, plot_title="Training Metrics", prefix="source"):
    """
    epochs     : list of epoch indices [1..N]
    acc_list   : list of accuracies per epoch
    loss_list  : list of losses per epoch (can be average training loss or validation loss, up to you)
    out_dir    : directory to save the plot
    plot_title : title for the plot
    prefix     : prefix in the saved plot name (e.g. 'source' or 'target')

    Produces a single figure with two subplots: accuracy vs. epoch and loss vs. epoch,
    then saves it as '{prefix}_training_plot.png' in out_dir.
    """
    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Accuracy
    axes[0].plot(epochs, acc_list, marker='o', color='blue', label='Accuracy')
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True)
    axes[0].legend()

    # Subplot 2: Loss
    axes[1].plot(epochs, loss_list, marker='o', color='red', label='Loss')
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    # Overall figure title
    fig.suptitle(plot_title, fontsize=14)

    # Adjust layout so titles, labels don’t overlap
    plt.tight_layout()

    # Save the figure
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{prefix}_training_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved to: {plot_path}")