import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import AUDIO_DATA_PREFIX


class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=48000, target_sr=16000):
        """
        Args:
            root_dir: Directory with 'real' and 'fake' subfolders
            target_length: Number of samples to use (48000 = 3 seconds at 16kHz)
            target_sr: Target sample rate (16000 Hz)
        """
        self.root_dir = root_dir
        self.target_length = target_length
        self.target_sr = target_sr
        self.file_paths = []
        self.labels = []
        
        # Load file paths and labels
        # real = 1, fake = 0
        for label, subfolder in enumerate(['fake', 'real']):
            folder_path = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist")
                continue
            
            files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.flac', '.mp3'))]
            for file in files:
                self.file_paths.append(os.path.join(folder_path, file))
                self.labels.append(label)
        
        print(f"Loaded {len(self.file_paths)} files from {root_dir}")
        print(f"  - Fake: {self.labels.count(0)}")
        print(f"  - Real: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Take first 3 seconds (48000 samples)
        waveform = waveform.squeeze(0)  # Remove channel dimension
        if waveform.shape[0] > self.target_length:
            waveform = waveform[:self.target_length]
        elif waveform.shape[0] < self.target_length:
            # Pad with zeros if shorter than target length
            padding = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform, label

def getRawNetLoaders(num_workers=4):
    trainDir = AUDIO_DATA_PREFIX + '/train'
    test1Dir = AUDIO_DATA_PREFIX + '/test_1'
    test2Dir = AUDIO_DATA_PREFIX + '/test_2'
    train_dataset = AudioDataset(trainDir, target_length=48000, target_sr=16000)
    test1_dataset = AudioDataset(test1Dir, target_length=48000, target_sr=16000)
    test2_dataset = AudioDataset(test2Dir, target_length=48000, target_sr=16000)
        
    trainLoader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test1Loader = DataLoader(
        test1_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test2Loader = DataLoader(
        test2_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainLoader, test1Loader, test2Loader