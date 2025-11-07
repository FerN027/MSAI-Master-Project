import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os

import config

class FFDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        self.class_to_idx = {}
        
        # Get all class folders
        classes = sorted([d for d in os.listdir(root_dir) 
                        if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # Collect images with train/test split
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all PNG images sorted by name
            images = sorted([f for f in os.listdir(class_dir) 
                           if f.lower().endswith('.png')])
            
            # Split: every 5 images -> first 4 for train, 5th for test
            for i, img_name in enumerate(images):
                img_path = os.path.join(class_dir, img_name)
                
                # Determine if this image is for training or testing
                # Position in group of 5: 0,1,2,3 -> train, 4 -> test
                position_in_group = i % 5
                
                if is_train and position_in_group < 4:
                    # Training image (first 4 of every 5)
                    self.samples.append((img_path, class_idx))
                elif not is_train and position_in_group == 4:
                    # Testing image (5th of every 5)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label






def imageFFLoaders():
    """
    Create train and test loaders for FaceForensics++ dataset.
    - Images are 128x128 resolution
    - Split: 4 images for training, 1 for testing (every 5 images)
    - First 4 images go to training, 5th image goes to testing
    - Training: random crop with padding + random horizontal flip + normalization
    - Testing: only normalization
    """
    root_dir = config.DATA_DIR
    
    # Training transformations: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(128, padding=8),  # Random crop with 8-pixel padding (standard for 128x128)
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standard normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test transformations: only normalization (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standard normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create train and test datasets
    train_dataset = FFDataset(root_dir, transform=train_transform, is_train=True)
    test_dataset = FFDataset(root_dir, transform=test_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"\nClass mapping (Folder -> Class Number):")
    for folder_name, class_idx in sorted(train_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {folder_name}: {class_idx}")
    
    return train_loader, test_loader










def mockingLoaders():
    """
    Create train and test loaders using CIFAR-10 dataset for testing the training pipeline.
    CIFAR-10 has 10 classes with 32x32 images (will be resized to fit XceptionNet).
    """
    # Training transformations: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Resize to XceptionNet standard input size
        transforms.RandomHorizontalFlip(),  # Horizontal flip augmentation
        transforms.RandomCrop(100, padding=4),  # Random crop augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standard normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test transformations: only resize + normalization (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    trainloader, _ = imageFFLoaders()
    # print total number of images
    print(f"Total training images: {len(trainloader.dataset)}")