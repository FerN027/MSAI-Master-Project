import numpy as np
import matplotlib.pyplot as plt

import config
from audio_loader_rawnet import getRawNetLoaders
from audio_loader_SpecRNet import getSpecRNetLoaders


def plot(modelNames=config.MODEL_NAMES, title=config.TITLE, folder="results/"):
    """
    Plot training and testing loss curves for multiple models.
    
    Assume each .npz has the same name as the model, and is stored in the specified folder:
        - two keys: 'train_loss' and 'test_loss'

    Choose color for each model (at most 3), and solid line for training loss, dashed line for testing loss.

    Plot a single figure with all training and testing loss curves for comparison:
        - legend should be properly placed on the figure;
        - title should be set according to the input argument.
    """
    # Define 3 clear and obvious colors
    colors = ['blue', 'green', 'red']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot each model
    for idx, model_name in enumerate(modelNames):
        # Load the .npz file
        file_path = f"{folder}{model_name}.npz"
        data = np.load(file_path)
        
        train_loss = data['train_loss']
        test_loss = data['test_loss']
        
        # Get color for this model
        color = colors[idx % len(colors)]
        
        # Plot training loss (solid line)
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, color=color, linestyle='-', linewidth=2)
        
        # Plot testing loss (dashed line)
        plt.plot(epochs, test_loss, color=color, linestyle='--', linewidth=2)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # First: line style indicators (train/test)
    legend_handles = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Train'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Test')
    ]
    legend_labels = ['Train', 'Test']
    
    # Then: colored squares for each model
    for idx, model_name in enumerate(modelNames):
        color = colors[idx % len(colors)]
        legend_handles.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor=color, markersize=16, label=model_name))
        legend_labels.append(model_name)
    
    # Configure plot
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=18, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save only (no show)
    plt.savefig(f'{folder}loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"Plot saved to {folder}loss_curves.png")

def audio_model_initialization(name: str, num_epoch: int):
    if name == 'RawNet2':
        from model_audio_1 import initializeComponent

    elif name == 'SpecRNet':
        from model_audio_2 import initializeComponent

    else:
        raise ValueError(f"Model {name} not recognized for initialization.")

    return initializeComponent(num_epoch)

def audio_loader_selection(name):
    if name == 'RawNet2':
        return getRawNetLoaders()

    elif name == 'SpecRNet':
        return getSpecRNetLoaders()

    else:
        raise ValueError(f"Model {name} not recognized for data loading.")


if __name__ == "__main__":
    plot()