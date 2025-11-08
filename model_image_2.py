import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16

import config

def get_vit_b_16(num_classes, image_size):
    """
    Get ViT-B/16 from scratch.
    
    Args:
        num_classes: Number of output classes;
        image_size: Input image size. Must be divisible by 16.
    """
    # Validate input size
    if image_size % 16 != 0:
        raise ValueError(f"Image size {image_size} must be divisible by 16 (patch size)")
    
    # Create ViT-B/16 from scratch
    model = vit_b_16(weights=None, image_size=image_size)
    
    # Modify the classifier head according to the number of classes
    model.heads = nn.Sequential(
        nn.Linear(768, num_classes)
    )
    
    return model

def initializeComponents(num_classes, image_size):
    print("Initializing ViT-B/16 ......")
    model = get_vit_b_16(num_classes, image_size)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    
    print("Done.")

    return model, optimizer, scheduler
