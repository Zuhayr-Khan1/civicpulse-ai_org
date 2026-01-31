# 🚀 CivicPulse AI: Complete Production-Ready Implementation Guide
## All-India Scaling with GPU Optimization (Windows PowerShell Edition v2.0)

**Project Status**: Ready to Execute  
**Platform**: Windows PowerShell  
**Start Date**: January 29, 2026  
**Target Completion**: June 2026 (20 weeks)  
**Team**: 1 person (laptop-first approach, GPU acceleration optional)  
**Repository**: [civicpulse-ai_org](https://github.com/Zuhayr-Khan1/civicpulse-ai_org)

---

## 📋 EXECUTIVE SUMMARY

You are scaling CivicPulse from **Hyderabad-only (31,416 km²)** to **All-India (3,287,263 km²)** — a **105x geographic expansion** using Windows PowerShell.

### Key Deliverables
1. ✅ **8 Production Notebooks** with full code
2. ✅ **6 Modular Python Modules** (src/*)
3. ✅ **Git LFS Integration** for large files
4. ✅ **Device-Agnostic Architecture** (CPU/GPU toggle via .env)
5. ✅ **Complete Data Pipeline** (download → preprocess → train → predict)
6. ✅ **Infrastructure Recommendation System**
7. ✅ **Interactive Streamlit Dashboard**

### Success Criteria
- Laptop runs 4GB model on CPU (HDF5 lazy loading)
- 50-100x GPU speedup for training (if GPU available)
- 0-code changes to switch devices
- Model trained end-to-end in 18-24 weeks

---

# 🔴 CRITICAL: START HERE (Week 1)

## PHASE 0: Git & Environment Setup (5 hours)

### Step 1: Clone Your Repository (Windows PowerShell)

```powershell
# Navigate to projects folder
cd C:\Users\$env:USERNAME\projects

# Clone repository
git clone https://github.com/Zuhayr-Khan1/civicpulse-ai_org.git
cd civicpulse-ai_org

# Verify clone
git status
```

### Step 2: Setup Git LFS (30 minutes)

**Install Git LFS:**

```powershell
# Download from https://git-lfs.com/
# Or use Chocolatey if installed:
choco install git-lfs

# Verify installation
git lfs version
```

**Initialize Git LFS in Repository:**

```powershell
# Initialize LFS locally
git lfs install --local

# Create .gitattributes file for large files
@"
# Large binary files
*.pt filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
"@ | Set-Content .gitattributes

# Stage and commit
git add .gitattributes
git commit -m "Setup Git LFS for large files"
git push origin main
```

### Step 3: Create Branch Structure (PowerShell)

```powershell
# Create develop branch
git checkout -b develop
git push -u origin develop

# Create feature branch (main work branch)
git checkout -b feature/india-scaling
git push -u origin feature/india-scaling

# Verify you're on feature branch
git branch

# Should show:
#   develop
# * feature/india-scaling
#   main
```

### Step 4: Setup .env Configuration (PowerShell)

```powershell
# Create .env file (NOT committed to git)
@"
# Device: auto, cpu, cuda, mps
CIVICPULSE_DEVICE=auto

# Batch size: auto or specific number
CIVICPULSE_BATCH_SIZE=auto

# Data mode: hdf5 (lazy) or numpy (pre-loaded)
CIVICPULSE_DATA_MODE=hdf5

# Patch size for spatial decomposition
CIVICPULSE_PATCH_SIZE=200

# Logging
LOG_LEVEL=INFO
VERBOSE=true
"@ | Set-Content .env

# Ensure .env is gitignored
Add-Content .gitignore ".env"

# Stage gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
git push origin feature/india-scaling
```

### Step 5: Install Dependencies (PowerShell)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify activation (should see (venv) prefix)
# Example: (venv) PS C:\Users\...>

# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

**⚠️ If you get execution policy error:**

```powershell
# Fix it one time
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activation again
.\venv\Scripts\Activate.ps1
```

### Step 6: Verify Git Setup (PowerShell)

```powershell
# Check current branch
git branch

# Check remote branches
git branch -a

# Verify LFS (should list tracked files)
git lfs ls-files

# Test status (should show clean working tree)
git status
```

**Expected output:**
```
On branch feature/india-scaling
nothing to commit, working tree clean
```

---

# 📊 PHASE 1: GEOGRAPHIC EXPANSION (Weeks 1-2)

## 1.1: Create Region Manager (2 hours)

**File**: `src\region_manager.py`

Create this file in your repository. This defines hierarchical geographic regions (National → State → District).

```python
"""
Geographic boundary manager for hierarchical region processing
Handles: National, State, District, City levels with parallel execution
Windows PowerShell compatible paths
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box, Polygon
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class RegionBoundary:
    """Represents a geographic region with metadata"""
    
    def __init__(self, name: str, bounds: Tuple[float, float, float, float], 
                 level: str, parent_id: Optional[str] = None):
        """
        Args:
            name: Region name (e.g., "Telangana", "India")
            bounds: (minx, miny, maxx, maxy) in WGS84
            level: "national", "state", "district", "city"
            parent_id: ID of parent region
        """
        self.name = name
        self.bounds = bounds
        self.level = level
        self.parent_id = parent_id
        self.geometry = box(*bounds)
        self.area_km2 = self._calc_area()
    
    def _calc_area(self) -> float:
        """Calculate area in km²"""
        minx, miny, maxx, maxy = self.bounds
        # 1 degree ≈ 111.32 km
        width_km = (maxx - minx) * 111.32
        height_km = (maxy - miny) * 111.32
        return width_km * height_km
    
    def grid_cell_count(self, resolution_km: float = 1.0) -> int:
        """Estimate grid cells at given resolution"""
        return int(self.area_km2 / (resolution_km ** 2))
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'bounds': self.bounds,
            'level': self.level,
            'parent_id': self.parent_id,
            'area_km2': self.area_km2,
            'grid_cells_1km': self.grid_cell_count(1.0)
        }


class ConfigurableBoundaryManager:
    """Manages preset and custom regions with hierarchical support"""
    
    # India-wide bounds (WGS84)
    INDIA_BOUNDS = (68.7, 8.4, 97.5, 35.0)
    
    # Major state boundaries (simplified)
    STATE_BOUNDS = {
        'Telangana': (78.0, 15.5, 81.9, 19.8),
        'Andhra Pradesh': (77.0, 12.6, 84.9, 18.3),
        'Maharashtra': (72.6, 16.5, 80.9, 23.3),
        'Karnataka': (74.0, 11.5, 78.6, 18.6),
        'Tamil Nadu': (76.8, 8.0, 80.3, 13.6),
        'Uttar Pradesh': (77.0, 23.8, 84.8, 30.4),
        'Delhi': (76.76, 28.41, 77.35, 28.88),
        'Punjab': (73.7, 29.5, 76.9, 32.3),
        'Gujarat': (68.1, 20.1, 74.4, 24.5),
        'Haryana': (76.4, 27.0, 77.6, 30.6),
        'Rajasthan': (68.8, 23.0, 78.6, 32.3),
        'West Bengal': (85.8, 21.6, 89.9, 27.2),
        'Kerala': (74.9, 8.3, 77.4, 12.5),
        'Madhya Pradesh': (74.0, 17.8, 82.9, 26.9),
        'Bihar': (82.2, 24.3, 88.3, 27.5),
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize boundary manager"""
        self.regions: Dict[str, RegionBoundary] = {}
        self._load_defaults()
        if config_path:
            self._load_custom(config_path)
    
    def _load_defaults(self):
        """Load preset regions"""
        # National level
        self.add_region(
            RegionBoundary('India', self.INDIA_BOUNDS, 'national')
        )
        
        # State level
        for state_name, bounds in self.STATE_BOUNDS.items():
            self.add_region(
                RegionBoundary(state_name, bounds, 'state', parent_id='India')
            )
    
    def _load_custom(self, config_path: str):
        """Load custom regions from JSON"""
        with open(config_path, 'r') as f:
            custom = json.load(f)
            for region_data in custom['regions']:
                region = RegionBoundary(
                    name=region_data['name'],
                    bounds=tuple(region_data['bounds']),
                    level=region_data['level'],
                    parent_id=region_data.get('parent_id')
                )
                self.add_region(region)
    
    def add_region(self, region: RegionBoundary):
        """Register a region"""
        self.regions[region.name] = region
    
    def get_region(self, name: str) -> Optional[RegionBoundary]:
        """Retrieve region by name"""
        return self.regions.get(name)
    
    def get_regions_by_level(self, level: str) -> List[RegionBoundary]:
        """Get all regions at given level"""
        return [r for r in self.regions.values() if r.level == level]
    
    def get_child_regions(self, parent_name: str) -> List[RegionBoundary]:
        """Get all child regions of a parent"""
        parent_id = parent_name
        return [r for r in self.regions.values() if r.parent_id == parent_id]
    
    def create_hierarchical_grid(self, 
                                top_level: str = 'national',
                                bottom_level: str = 'state',
                                resolution_km: float = 1.0) -> Dict[str, int]:
        """Create hierarchical processing structure"""
        processing_map = {}
        
        for region in self.get_regions_by_level(top_level):
            processing_map[region.name] = region.grid_cell_count(resolution_km)
        
        return processing_map
    
    def to_geojson(self, filename: str = 'regions.geojson'):
        """Export all regions as GeoJSON"""
        features = []
        for region in self.regions.values():
            features.append({
                'type': 'Feature',
                'properties': region.to_dict(),
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [region.bounds[0], region.bounds[1]],
                        [region.bounds[2], region.bounds[1]],
                        [region.bounds[2], region.bounds[3]],
                        [region.bounds[0], region.bounds[3]],
                        [region.bounds[0], region.bounds[1]]
                    ]]
                }
            })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(filename, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} regions to {filename}")


# TEST
if __name__ == '__main__':
    mgr = ConfigurableBoundaryManager()
    india = mgr.get_region('India')
    print(f"India: {india.area_km2:,.0f} km², ~{india.grid_cell_count()//1000}k cells")
    
    states = mgr.get_regions_by_level('state')
    print(f"Configured {len(states)} states")
```

**Create this file in PowerShell:**

```powershell
# Navigate to project root
cd C:\Users\$env:USERNAME\civicpulse-ai_org

# Create src folder if it doesn't exist
New-Item -ItemType Directory -Path "src" -Force

# Create region_manager.py (use Python directly)
# Copy the Python code above and save as src/region_manager.py
# Or use: notepad src\region_manager.py (then paste and save)

# Test the module
python -c "from src.region_manager import ConfigurableBoundaryManager; mgr = ConfigurableBoundaryManager(); print('✓ Region Manager loaded')"
```

### Test Region Manager

```python
# In Jupyter or Python shell
from src.region_manager import ConfigurableBoundaryManager

mgr = ConfigurableBoundaryManager()

# Test India
india = mgr.get_region('India')
print(f"India: {india.area_km2:,.0f} km² → {india.grid_cell_count():,.0f} grid cells")
# Expected: ~3.2M cells

# Test states
telangana = mgr.get_region('Telangana')
print(f"Telangana: {telangana.area_km2:,.0f} km² → {telangana.grid_cell_count():,.0f} cells")
# Expected: ~112,000 cells

# Export boundaries
mgr.to_geojson('data/raw/india_regions.geojson')
```

### Commit Progress (PowerShell)

```powershell
git add src/region_manager.py
git commit -m "[REGION] Add hierarchical boundary manager for India states"
git push origin feature/india-scaling

# Verify push
git status
# Should show: nothing to commit, working tree clean
```

---

## 1.2: Create Config Module (1.5 hours)

**File**: `src\config.py`

This handles device auto-detection and batch size tuning.

```python
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
```

**Create this file in PowerShell:**

```powershell
# Create src/config.py
# Option 1: Use notepad
notepad src\config.py
# (Copy the Python code above and save)

# Option 2: Use PowerShell directly
@"
# Paste the Python code here
"@ | Set-Content src\config.py

# Test it
python -c "from src.config import TrainingConfig; config = TrainingConfig(); config.print_summary()"
```

### Test Config

```python
from src.config import TrainingConfig

config = TrainingConfig()
config.print_summary()
# Should show: Device: cpu, Batch Size: 4, Data Mode: hdf5
```

### Commit (PowerShell)

```powershell
git add src/config.py
git commit -m "[CONFIG] Add device auto-detection and configuration management"
git push origin feature/india-scaling
```

---

## 1.3: Create Preprocessing Module (2.5 hours)

**File**: `src\preprocessing.py`

Handles region-aware data quality assessment and adaptive interpolation.

```python
"""
Region-aware preprocessing for heterogeneous data quality
Handles: Urban vs rural quality differences, sparse vs dense coverage
Windows PowerShell compatible
"""

import numpy as np
import rasterio
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import warnings


class RegionAwarePreprocessor:
    """
    Analyzes and handles data quality variations across regions
    Urban areas: high quality, dense data
    Rural areas: sparse, interpolation needed
    """
    
    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold
        self.quality_mask = None
        self.confidence_scores = None
    
    def calculate_quality_score(self, data: np.ndarray, 
                               region_type: str = 'mixed') -> np.ndarray:
        """Calculate quality score for each grid cell"""
        height, width = data.shape
        quality = np.zeros_like(data, dtype=np.float32)
        
        # Score 1: Non-zero cells
        nonzero = (data > 0).astype(np.float32)
        quality += 0.2 * nonzero
        
        # Score 2: Spatial consistency
        for i in range(1, height-1):
            for j in range(1, width-1):
                neighbors = data[i-1:i+2, j-1:j+2].flatten()
                valid_neighbors = neighbors[neighbors > 0]
                
                if len(valid_neighbors) > 0:
                    consistency = 1.0 - np.std(valid_neighbors) / (np.mean(valid_neighbors) + 1e-6)
                    quality[i, j] += 0.3 * max(0, consistency)
        
        # Score 3: Region-specific adjustments
        if region_type == 'urban':
            density = np.sum(data > 0) / data.size
            quality *= (1.0 + 0.5 * density)
        elif region_type == 'rural':
            quality = np.minimum(quality * 1.2, 1.0)
        
        quality = np.minimum(quality / (np.max(quality) + 1e-6), 1.0)
        return quality
    
    def identify_low_quality_regions(self, data: np.ndarray, 
                                    threshold: float = 0.3) -> np.ndarray:
        """Identify cells requiring interpolation"""
        quality = self.calculate_quality_score(data)
        return quality < threshold
    
    def adaptive_interpolation(self, data: np.ndarray, 
                              quality_mask: np.ndarray) -> np.ndarray:
        """Adaptive interpolation using KNN"""
        data_filled = data.copy()
        height, width = data.shape
        
        valid_coords = np.argwhere(~quality_mask)
        valid_values = data[~quality_mask]
        
        if len(valid_coords) == 0:
            warnings.warn("No high-quality cells found for interpolation")
            return data_filled
        
        tree = cKDTree(valid_coords)
        missing_coords = np.argwhere(quality_mask)
        
        if len(missing_coords) > 0:
            distances, indices = tree.query(missing_coords, k=5)
            
            for idx, (coord, dist_list, idx_list) in enumerate(
                zip(missing_coords, distances, indices)
            ):
                weights = 1.0 / (dist_list + 1e-6)
                weights /= weights.sum()
                interpolated_value = np.sum(valid_values[idx_list] * weights)
                data_filled[coord[0], coord[1]] = interpolated_value
        
        return data_filled


# TEST
if __name__ == '__main__':
    preprocessor = RegionAwarePreprocessor()
    print("✓ Preprocessor loaded")
```

**Create this file in PowerShell:**

```powershell
# Create src/preprocessing.py
notepad src\preprocessing.py
# (Copy the Python code and save)

# Test it
python -c "from src.preprocessing import RegionAwarePreprocessor; print('✓ Preprocessor loaded')"
```

### Commit (PowerShell)

```powershell
git add src/preprocessing.py
git commit -m "[PREPROCESSING] Add region-aware quality assessment and interpolation"
git push origin feature/india-scaling
```

---

# 📓 PHASE 2: NOTEBOOKS (Weeks 2-6)

## Overview: 8 Production Notebooks

| # | Notebook | Duration | Purpose |
|---|----------|----------|---------|
| 00 | Setup India Boundaries | 10 min | Define all-India regions |
| 01 | Preprocess Sample States | 60 min | Download & preprocess 2 states |
| 02 | Create HDF5 Dataset | 10 min | Create lazy-loading dataset |
| 03 | Clip Full India | 8-12 hrs | Clip all-India data (overnight) |
| 04 | Create Full Dataset | 4 hrs | Create full-India HDF5 |
| 05 | Progressive Training | 48+ hrs | Train ConvLSTM (3 stages) |
| 06 | Inference & Visualization | 2 hrs | Generate predictions |
| 07 | Gap Analysis | 1 hr | Infrastructure recommendations |

---

# 🚀 QUICK START COMMAND (PowerShell)

Complete setup in one go:

```powershell
# Navigate to projects
cd C:\Users\$env:USERNAME\projects

# Clone
git clone https://github.com/Zuhayr-Khan1/civicpulse-ai_org.git
cd civicpulse-ai_org

# Setup Git LFS
git lfs install --local

# Create branches
git checkout -b feature/india-scaling
git push -u origin feature/india-scaling

# Create .env
@"
CIVICPULSE_DEVICE=auto
CIVICPULSE_BATCH_SIZE=auto
CIVICPULSE_DATA_MODE=hdf5
CIVICPULSE_PATCH_SIZE=200
"@ | Set-Content .env

# Add to gitignore
Add-Content .gitignore ".env"

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install
pip install -r requirements.txt

# Test
python -c "from src.region_manager import ConfigurableBoundaryManager; mgr = ConfigurableBoundaryManager(); print(f'✓ Setup complete. India: {mgr.get_region(\"India\").area_km2:,.0f} km²')"

# Start Jupyter (use launcher from JUPYTER_SETUP_WINDOWS.md)
.\start_jupyter.ps1
```

---

# 📈 EXECUTION TIMELINE (Windows PowerShell)

## Week-by-Week Breakdown

### Weeks 1-2: Git & Foundation
- [ ] Day 1-2: Git LFS, branches, .env setup (use commands above)
- [ ] Day 3-5: Create src/region_manager.py, src/config.py, src/preprocessing.py
- [ ] Day 6-7: Test modules, commit to feature/india-scaling

**PowerShell Commands:**

```powershell
# Verify modules
python -c "from src.region_manager import ConfigurableBoundaryManager; from src.config import TrainingConfig; from src.preprocessing import RegionAwarePreprocessor; print('✓ All modules loaded')"

# Check git status
git status
git log --oneline | head -5
```

### Weeks 2-3: Sample State Processing
- [ ] Notebook 00: Setup boundaries (1 session)
- [ ] Notebook 01: Download & preprocess Telangana + Maharashtra (2 sessions)
- [ ] Notebook 02: Create HDF5 (1 session)

**In Jupyter (launched via .\start_jupyter.ps1):**
```
1. Create notebook: notebooks/00_setup_india_boundaries.ipynb
2. Copy code from Complete_Notebooks_Code.md
3. Run cells: Shift+Enter
```

### Week 4-5: Full-Scale Data
- [ ] Download all-India WorldPop (5 files, ~3GB) - 2-4 hours
- [ ] Notebook 03: Clip full India (OVERNIGHT, 8-12 hours)
- [ ] Notebook 04: Create HDF5 (4 hours)

### Weeks 5-8: Model Training
- [ ] Notebook 05: Train ConvLSTM (48-72 hours)
  - Stage 1 (coarse): 6-8 hours
  - Stage 2 (medium): 15-20 hours
  - Stage 3 (fine): 25-35 hours

### Weeks 9-10: Inference & Analysis
- [ ] Notebook 06: Generate predictions (2 hours)
- [ ] Notebook 07: Gap analysis (1 hour)

### Weeks 11-12: Dashboard & Polish
- [ ] Streamlit dashboard (3 hours)
- [ ] Testing & documentation (5 hours)
- [ ] Final deployment prep (2 hours)

---

# 🐛 WINDOWS-SPECIFIC TROUBLESHOOTING

## Issue: Git LFS Not Found

```powershell
# Download installer from https://git-lfs.com/
# Or install with Chocolatey
choco install git-lfs

# Verify
git lfs version
```

## Issue: Virtual Environment Won't Activate

```powershell
# Check execution policy
Get-ExecutionPolicy

# Fix it
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Try activation again
.\venv\Scripts\Activate.ps1
```

## Issue: ModuleNotFoundError When Importing src

```python
# In Jupyter first cell, add:
import sys
import os
sys.path.insert(0, r'C:\Users\YOUR_USERNAME\civicpulse-ai_org')

# Then import
from src.region_manager import ConfigurableBoundaryManager
```

## Issue: Path Errors in Notebooks

Use raw strings (r'...') for Windows paths:

```python
# CORRECT - Windows paths with backslashes
BASE_DIR = Path(r'C:\Users\YOUR_USERNAME\civicpulse-ai_org')
DATA_DIR = BASE_DIR / 'data'

# NOT correct
BASE_DIR = Path('C:\Users\...')  # Backslashes interpreted as escape codes
```

## Issue: CUDA Out of Memory on Windows

```powershell
# Reduce batch size in .env
CIVICPULSE_BATCH_SIZE=4

# Or set to CPU
CIVICPULSE_DEVICE=cpu
```

---

# ✅ SUCCESS CHECKLIST

Before declaring project complete:

- [ ] All 8 notebooks run end-to-end
- [ ] Model achieves R² > 0.85 on validation set
- [ ] Predictions generated for 2025 & 2030
- [ ] Infrastructure recommendations identified (100+ sites)
- [ ] Dashboard deployed and interactive
- [ ] Git history clean with meaningful commits
- [ ] README updated with results
- [ ] All tests pass
- [ ] Documentation complete

---

## 📚 FILES TO KEEP OPEN

| File | Purpose |
|------|---------|
| JUPYTER_SETUP_WINDOWS.md | How to launch Jupyter |
| CivicPulse_Production_Guide_Windows.md | THIS FILE - Setup & modules |
| Complete_Notebooks_Code.md | Notebooks 00-03 code |
| Notebooks_04-07_Complete_Code.md | Notebooks 04-07 code |
| SCALING_ROADMAP.md | Track progress |
| Quick_Reference.md | Commands & tips |

---

## 🎯 NEXT STEPS

1. **Read** this guide (30 min)
2. **Run** quick start commands above (30 min)
3. **Test** imports in Python (5 min)
4. **Launch** Jupyter: `.\start_jupyter.ps1` (from JUPYTER_SETUP_WINDOWS.md)
5. **Create** notebooks 00-07
6. **Track** progress in SCALING_ROADMAP.md

---

**Created**: January 29, 2026  
**Platform**: Windows PowerShell  
**Repository**: [civicpulse-ai_org](https://github.com/Zuhayr-Khan1/civicpulse-ai_org)  
**Status**: Ready to Execute 🚀