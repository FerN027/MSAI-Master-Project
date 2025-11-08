import torch.optim as optim
from torchvision.models import maxvit_t

import config

def get_maxvit_t(num_classes):
    """
    Get MaxViT-T (Tiny) model from torchvision.
    
    MaxViT is a hybrid architecture that combines:
    - MBConv blocks (Mobile Inverted Bottleneck Convolution)
    - Multi-axis attention (Block attention + Grid attention)
    
    Each MaxViT block = MBConv → Block Attention → Grid Attention → FFN
    """
    model = maxvit_t(weights=None, num_classes=num_classes)
    
    return model

def initializeComponents(num_classes):
    print("Initializing MaxViT-T ......")
    model = get_maxvit_t(num_classes)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
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
