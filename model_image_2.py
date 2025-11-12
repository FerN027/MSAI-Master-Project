import torch.optim as optim
from timm import create_model

import config


def get_vit_tiny(num_classes, image_size):
    """
    Get ViT-Tiny model.
    
    Architecture:
    - Patch size: 16x16
    - Hidden dimension: 192
    - Number of layers: 12
    - Number of heads: 3
    - MLP ratio: 4
    - Parameters: ~5.7M
    """
    model = create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=image_size
    )
    
    return model

def initializeComponents(num_classes, image_size):
    print("Initializing ViT-Tiny ...")
    model = get_vit_tiny(num_classes, image_size)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    warmup_epochs = 15
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / 3e-4,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS - warmup_epochs,
        eta_min=1e-6
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    print("Done.")

    return model, optimizer, scheduler
