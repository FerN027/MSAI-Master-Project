# MSAI Master Project: Deepfake Detection Across Image and Audio Modalities

This repository contains the source code, experimental pipelines, and dissertation assets for a master's project exploring deepfake creation methods and detection approaches across image and audio modalities.

## 📖 Project Overview

The project provides a comparative experimental evaluation of different architectural paradigms for detecting both visual and acoustic spoofing/forgeries. 

### 🖼️ Image Forgery Detection
Focuses on evaluating and comparing CNN-based, Transformer-based, and Hybrid architectures to detect manipulated facial content. 
- **Dataset**: [FaceForensics++](https://github.com/ondyari/FaceForensics) (Categorized into Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, and Real)
- **Evaluated Architectures**:
  - **XceptionNet** (`model_image_1.py`): A purely Convolutional approach.
  - **ViT-Tiny** (`model_image_2.py`): A purely Vision Transformer (attention-based) approach.
  - **MobileViT-S** (`model_image_3.py`): A hybrid architecture combining the strengths of CNNs and Transformers.

### 🎙️ Audio Spoofing Detection
Focuses on exploring methods to detect synthetically generated or spoofed speech.
- **Dataset**: [ASVspoof 2019 LA - Logical Access](https://www.asvspoof.org)
- **Evaluated Architectures**:
  - **RawNet2** (`model_audio_1.py` / `audio_loader_rawnet.py`): Processes raw audio waveforms directly.
  - **SpecRNet** (`model_audio_2.py` / `audio_loader_SpecRNet.py`): Operates on acoustic spectrograms.

## 📂 Repository Structure

```text
├── main.py                    # Entry point for training/testing image forgery models
├── trainer.py                 # Core training loop and evaluation logic for image models
├── dataloader.py              # Data loaders and transforms for the FaceForensics++ dataset
├── model_image_{1,2,3}.py     # PyTorch implementations of XceptionNet, ViT, and MobileViT
├── audio_main.py              # Entry point for training/testing audio spoofing models
├── audio_train.py             # Core training loop for audio models
├── audio_loader_*.py          # Audio-specific data loading (Raw form and Spectrograms)
├── model_audio_{1,2}.py       # PyTorch implementations of RawNet and SpecRNet architectures
├── config.py                  # Global hyperparameters (Epochs, Dataset Paths)
├── helper.py / utility.py     # Utility functions and shared helper logic
├── data/                      # Directory for datasets (ASVspoof_LA, FF_faces_simplified)
├── results/                   # Evaluation results, loss curves (.npz), and model weights
└── LaTeX/                     # Master's dissertation source code and compiled figures
```

## 🚀 Getting Started

### Prerequisites
The project uses **PyTorch** for its deep learning framework. You will need to install standard data science and deep learning packages:
```bash
pip install torch torchvision torchaudio numpy pandas librosa
```

### Running the Code

**1. Configuration**
Update the dataset pathways in `config.py`:
```python
DATA_DIR = "data/FF_faces_simplified"   # Image dataset path
AUDIO_DATA_PREFIX = "data/ASVspoof_LA"  # Audio dataset path
```

**2. Image Models**
To train or evaluate image models, adjust the initializations in `main.py` and run:
```bash
python main.py
```

**3. Audio Models**
To train or evaluate audio models, adjust the model selection in `audio_main.py` and run:
```bash
python audio_main.py
```

## 📄 Documentation
The formal writeup and theoretical evaluations of this project are located in the `LaTeX/` directory, structured into chapters (e.g., `Chapter3/3.1 image.tex` and `3.2 audio.tex`, etc.). The abstract and conclusions detail how hybrid architectures balance out various trade-offs between generalization and raw classification performance.
