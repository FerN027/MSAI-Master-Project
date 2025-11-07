import numpy as np
import matplotlib.pyplot as plt

import config

def plot(modelNames=config.MODEL_NAMES, title=config.TITLE, folder="results/"):
    # Define 3 clear and obvious colors
    colors = ['blue', 'green', 'red']
    
    """
    Assume each .npz has the same name as the model, and is stored in the specified folder:
        - two keys: 'train_loss' and 'test_loss'

    Choose color for each model (at most 3), and solid line for training loss, dashed line for testing loss.

    Plot a single figure with all training and testing loss curves for comparison:
        - legend should be properly placed on the figure;
        - title should be set according to the input argument.
    """