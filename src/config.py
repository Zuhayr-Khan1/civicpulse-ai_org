"""
Configuration management with device detection
Automatically selects optimal hardware (CPU/GPU) and batch sizes
Reads from .env for device-specific tuning
Windows PowerShell compatible
"""

import os
import torch
import sys
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


class DeviceConfig:
    """Hardware and batch size auto-detection"""
    
    @staticmethod
    def get_device(prefer_gpu: bool = True) -> torch.device:
        """Automatically detect best available device"""
        device_env = os.getenv('CIVICPULSE_DEVICE', 'auto')
        
        if device_env == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
        elif device_env == 'cpu':
            return torch.device('cpu')
        elif device_env == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
        elif device_env == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
        
        return torch.device('cpu')
    
    @staticmethod
    def get_vram_gb() -> float:
        """Get available GPU VRAM in GB"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory / 1e9
        return 0.0
    
    @staticmethod
    def get_optimal_batch_size(device: torch.device) -> int:
        """Recommend batch size based on available VRAM"""
        batch_env = os.getenv('CIVICPULSE_BATCH_SIZE', 'auto')
        if batch_env != 'auto':
            return int(batch_env)
        
        if device.type == 'cuda':
            vram_gb = DeviceConfig.get_vram_gb()
            
            if vram_gb > 40:
                return 64
            elif vram_gb > 20:
                return 32
            elif vram_gb > 12:
                return 16
            else:
                return 8
        
        return 4  # CPU mode
    
    @staticmethod
    def get_data_mode() -> str:
        """Get data loading mode: hdf5 (lazy) or numpy (pre-loaded)"""
        return os.getenv('CIVICPULSE_DATA_MODE', 'hdf5')
    
    @staticmethod
    def get_patch_size() -> int:
        """Get spatial patch size for decomposed processing"""
        return int(os.getenv('CIVICPULSE_PATCH_SIZE', '200'))


class TrainingConfig:
    """Training hyperparameters"""
    
    DEVICE = DeviceConfig.get_device()
    BATCH_SIZE = DeviceConfig.get_optimal_batch_size(DEVICE)
    VRAM_GB = DeviceConfig.get_vram_gb()
    
    # Model architecture
    IN_CHANNELS = 1
    NUM_LAYERS = 2
    HIDDEN_CHANNELS = 32
    KERNEL_SIZE = 3
    
    # Training
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    
    # Data
    SEQ_LENGTH = 4
    TRAIN_SPLIT = 0.8
    DATA_MODE = DeviceConfig.get_data_mode()
    PATCH_SIZE = DeviceConfig.get_patch_size()
    
    # Directories (Windows-compatible paths)
    CHECKPOINT_DIR = Path('models/checkpoints')
    LOG_DIR = Path('logs')
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("CIVICPULSE TRAINING CONFIGURATION")
        print("="*60)
        print(f"Device: {cls.DEVICE}")
        print(f"VRAM Available: {cls.VRAM_GB:.1f} GB")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Data Mode: {cls.DATA_MODE}")
        print(f"Patch Size: {cls.PATCH_SIZE}×{cls.PATCH_SIZE} cells")
        print("="*60 + "\n")


# TEST
if __name__ == '__main__':
    config = TrainingConfig()
    config.print_summary()