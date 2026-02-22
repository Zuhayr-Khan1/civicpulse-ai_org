""" 
Configuration management — benchmark-driven 
Reads ALL values from .env file set by 03-PERF_Performance_Benchmarker.ipynb 
No more hardcoded defaults or auto-guessing 
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class DeviceConfig:
    """Benchmark-driven configuration — all values from .env"""

    @staticmethod
    def get_device() -> torch.device:
        """Get device from CIVICPULSE_DEVICE in .env"""
        device_env = os.getenv('CIVICPULSE_DEVICE', 'cpu')  # safe default
        if device_env == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("⚠️  CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
        elif device_env == 'cpu':
            return torch.device('cpu')
        elif device_env == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("⚠️  MPS requested but not available, falling back to CPU")
                return torch.device('cpu')
        else:
            # Invalid value in .env
            raise ValueError(f"CIVICPULSE_DEVICE='{device_env}' invalid. Use: cpu, cuda, or mps")

    @staticmethod
    def get_vram_gb() -> float:
        """Get available GPU VRAM in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0

    @staticmethod
    def get_batch_size() -> int:
        """Get batch size from CIVICPULSE_BATCH_SIZE in .env (set by benchmark)"""
        batch_str = os.getenv('CIVICPULSE_BATCH_SIZE')
        if batch_str is None or batch_str.lower() == 'auto':
            raise ValueError(
                "CIVICPULSE_BATCH_SIZE not set in .env. "
                "Run 03-PERF_Performance_Benchmarker.ipynb first, "
                "then copy recommended value into .env"
            )
        try:
            return int(batch_str)
        except ValueError:
            raise ValueError(f"CIVICPULSE_BATCH_SIZE='{batch_str}' must be integer")

    @staticmethod
    def get_data_mode() -> str:
        """Get data mode from CIVICPULSE_DATA_MODE in .env (hdf5 or normal)"""
        return os.getenv('CIVICPULSE_DATA_MODE', 'hdf5')

    @staticmethod
    def get_patch_size() -> int:
        """Get patch size from CIVICPULSE_PATCH_SIZE in .env (set by benchmark)"""
        patch_str = os.getenv('CIVICPULSE_PATCH_SIZE', '200')
        try:
            return int(patch_str)
        except ValueError:
            raise ValueError(f"CIVICPULSE_PATCH_SIZE='{patch_str}' must be integer")

class TrainingConfig:
    """Training hyperparameters — all driven by .env from benchmark notebook"""

    DEVICE        = DeviceConfig.get_device()
    BATCH_SIZE    = DeviceConfig.get_batch_size()
    VRAM_GB       = DeviceConfig.get_vram_gb()

    # Model architecture — override these in .env if benchmark recommended changes
    # CIVICPULSE_HIDDEN_CHANNELS=64
    # CIVICPULSE_NUM_LAYERS=2
    HIDDEN_CHANNELS = int(os.getenv('CIVICPULSE_HIDDEN_CHANNELS', '64'))
    NUM_LAYERS      = int(os.getenv('CIVICPULSE_NUM_LAYERS', '2'))
    IN_CHANNELS     = 1
    KERNEL_SIZE     = 3

    # Training hyperparameters
    NUM_EPOCHS     = 100
    LEARNING_RATE  = 1e-3
    WEIGHT_DECAY   = 1e-5
    GRADIENT_CLIP  = 1.0

    # Data
    SEQ_LENGTH     = 4
    TRAIN_SPLIT    = 0.8
    DATA_MODE      = DeviceConfig.get_data_mode()
    PATCH_SIZE     = DeviceConfig.get_patch_size()

    # Directories (Windows-compatible paths)
    CHECKPOINT_DIR = Path('models/checkpoints')
    LOG_DIR        = Path('logs')

    @classmethod
    def print_summary(cls):
        """Print configuration summary"""

        print("\n" + "="*70)
        print("🚀 CIVICPULSE TRAINING CONFIGURATION (Benchmark-Optimized)")
        print("="*70)
        print(f"Device           : {cls.DEVICE}")
        print(f"VRAM Available   : {cls.VRAM_GB:.1f} GB")
        print(f"Batch Size       : {cls.BATCH_SIZE}")
        print(f"Data Mode        : {cls.DATA_MODE}")
        print(f"Patch Size       : {cls.PATCH_SIZE}×{cls.PATCH_SIZE} cells")
        print(f"ConvLSTM         : {cls.HIDDEN_CHANNELS} hidden, {cls.NUM_LAYERS} layers")
        print(f"Learning Rate    : {cls.LEARNING_RATE}")
        print(f"Sequence Length  : {cls.SEQ_LENGTH} timesteps")
        print("="*70 + "\n")

# TEST
if __name__ == '__main__':
    config = TrainingConfig()
    config.print_summary()
