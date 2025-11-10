import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import AUDIO_DATA_PREFIX


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, target_sr=16000, duration=4.0):
        """
        Args:
            root_dir: Directory with 'real' and 'fake' subfolders
            target_sr: Target sample rate (16000 Hz)
            duration: Audio duration in seconds (4.0 seconds)
        
        Standard spectrogram configuration to get shape (1, 80, 404):
        - Mel-spectrogram with 80 mel bins
        - n_fft: 512
        - win_length: 400 samples (25 ms at 16kHz)
        - hop_length: 160 samples (10 ms at 16kHz)
        - Input: 4 seconds at 16kHz = 64000 samples
        - Time frames: (64000 - 400) / 160 + 1 = 398 frames → pad to 404
        """
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration)  # 64000 samples
        self.file_paths = []
        self.labels = []
        
        # Standard mel-spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=512,
            win_length=400,  # 25 ms at 16kHz
            hop_length=160,  # 10 ms at 16kHz
            n_mels=80,
            f_min=0.0,
            f_max=target_sr / 2.0,
            power=2.0,
            window_fn=torch.hann_window,
            normalized=False
        )
        
        # Convert power spectrogram to dB scale (log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        
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
        
        # Load audio (supports .wav, .flac, .mp3)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if necessary
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Remove channel dimension
        waveform = waveform.squeeze(0)  # (num_samples,)
        
        # Normalize length to 4 seconds (64000 samples)
        if waveform.shape[0] > self.target_length:
            # If longer: take first 4 seconds
            waveform = waveform[:self.target_length]
        elif waveform.shape[0] < self.target_length:
            # If shorter: zero-pad to 4 seconds
            padding = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Add channel dimension for mel-spectrogram transform
        waveform = waveform.unsqueeze(0)  # (1, 64000)
        
        # Compute mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # (1, 80, ~398)
        
        # Convert to dB scale (log scale for better representation)
        mel_spec_db = self.amplitude_to_db(mel_spec)  # (1, 80, ~398)
        
        # Ensure exactly 404 time frames
        current_frames = mel_spec_db.shape[2]
        if current_frames > 404:
            # Crop to 404
            mel_spec_db = mel_spec_db[:, :, :404]
        elif current_frames < 404:
            # Zero-pad to 404
            pad_amount = 404 - current_frames
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad_amount))
        
        return mel_spec_db, label

def getSpecRNetLoaders(num_workers=4, batch_size=128):
    """
    Get DataLoaders for SpecRNet model.
    
    Returns:
        tuple: (trainLoader, test1Loader, test2Loader)
    """
    trainDir = AUDIO_DATA_PREFIX + '/train'
    test1Dir = AUDIO_DATA_PREFIX + '/test_1'
    test2Dir = AUDIO_DATA_PREFIX + '/test_2'
    
    train_dataset = SpectrogramDataset(trainDir, target_sr=16000, duration=4.0)
    test1_dataset = SpectrogramDataset(test1Dir, target_sr=16000, duration=4.0)
    test2_dataset = SpectrogramDataset(test2Dir, target_sr=16000, duration=4.0)
    
    trainLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test1Loader = DataLoader(
        test1_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test2Loader = DataLoader(
        test2_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return trainLoader, test1Loader, test2Loader
