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

def audio_plot():
    """
    Plot training and testing loss curves for audio models.
    
    Each .npz file has three keys: 'train', 'test_1', 'test_2'
    Uses two colors (ocean blue and orange) and three line styles (solid, dashed, dotted).
    """
    name_pool = ['RawNet2', 'SpecRNet']
    title = "Loss Curves on ASVspoof_LA"
    save_name = "audio_loss_curves.png"
    folder = "results/"
    
    # Define colors: ocean blue and orange
    colors = ['#006994', '#FF8C00']  # Ocean blue, Dark orange
    
    # Define line styles for train, test_1, test_2
    line_styles = {
        'train': '-',      # solid
        'test_1': '--',    # dashed
        'test_2': ':'      # dotted
    }
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot each model
    for idx, model_name in enumerate(name_pool):
        # Load the .npz file
        file_path = f"{folder}{model_name}_losses.npz"
        data = np.load(file_path)
        
        train_loss = data['train']
        test1_loss = data['test_1']
        test2_loss = data['test_2']
        
        # Get color for this model
        color = colors[idx % len(colors)]
        
        # Number of epochs
        epochs = np.arange(1, len(train_loss) + 1)
        
        # Plot training loss (solid line)
        plt.plot(epochs, train_loss, color=color, linestyle=line_styles['train'], 
                linewidth=2, label=f'{model_name} Train')
        
        # Plot test_1 loss (dashed line)
        plt.plot(epochs, test1_loss, color=color, linestyle=line_styles['test_1'], 
                linewidth=2, label=f'{model_name} Test1')
        
        # Plot test_2 loss (dotted line)
        plt.plot(epochs, test2_loss, color=color, linestyle=line_styles['test_2'], 
                linewidth=2, label=f'{model_name} Test2')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # First: line style indicators (train/test_1/test_2)
    legend_handles = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Train'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Test_1'),
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, label='Test_2')
    ]
    legend_labels = ['Train', 'Test_1', 'Test_2']
    
    # Then: colored squares for each model
    for idx, model_name in enumerate(name_pool):
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
    
    # Save plot
    save_path = f'{folder}{save_name}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"Audio loss curves saved to {save_path}")

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
    # plot()

    audio_plot()