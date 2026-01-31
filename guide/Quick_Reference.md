# CivicPulse India: Quick Reference & Execution Checklist

**Status:** Ready to Execute | **Date:** January 28, 2026 | **All-India Scale**

---

## 📋 FILES YOU NOW HAVE

| File | Purpose | Status |
|------|---------|--------|
| `CivicPulse_Production_Guide.md` | **START HERE** - Full implementation blueprint | ✅ Complete |
| `Complete_Notebooks_Code.md` | All 8 notebooks with full code ready to copy | ✅ Complete |
| `SCALING_ROADMAP.md` | Week-by-week execution plan | ✅ In Repo |
| `CivicPulse_India_Scale.md` | GPU optimization guide + architecture | ✅ In Repo |
| `git_migration_guide.md` | Git setup, branching, LFS | ✅ In Repo |
| `CivicPulse_Detailed_Guide_old-hyd-only.md` | Historical Hyderabad reference (FYI) | ℹ️ Legacy |

---

## 🚀 START HERE: First 2 Hours

### Step 1: Git Setup (30 minutes)
```bash
cd ~/civicpulse-ai  # Your existing repo
git lfs install
git checkout -b develop && git push -u origin develop
git checkout -b feature/india-scaling && git push -u origin feature/india-scaling
cat > .env << 'EOF'
CIVICPULSE_DEVICE=auto
CIVICPULSE_BATCH_SIZE=auto
CIVICPULSE_DATA_MODE=hdf5
CIVICPULSE_PATCH_SIZE=200
EOF
git add .gitattributes .env .gitignore
git commit -m "[SETUP] Initialize India-scale development environment"
```

### Step 2: Implement Core Modules (90 minutes)

**Copy these files to your `src/` directory:**

1. **`src/region_manager.py`** (from Production Guide)
   - RegionBoundary class + ConfigurableBoundaryManager
   - Tests on your Hyderabad region first

2. **Update `src/preprocessing.py`**
   - Add RegionAwarePreprocessor class
   - Add calculate_quality_score() method
   - Add adaptive_interpolation() method

3. **Update `src/config.py`**
   - Add device auto-detection (torch.cuda)
   - Add batch size tuning
   - Add .env variable loading

**Test everything:**
```python
from src.region_manager import ConfigurableBoundaryManager
from src.config import TrainingConfig

# Should work
mgr = ConfigurableBoundaryManager()
india = mgr.get_region('India')
print(f"✓ India: {india.area_km2:,.0f} km²")

config = TrainingConfig()
config.print_summary()
print(f"✓ Device: {config.DEVICE}, Batch: {config.BATCH_SIZE}")
```

---

## 📓 THE 8 NOTEBOOKS: Execution Order

| # | Name | Hours | Week | Status | Key Output |
|---|------|-------|------|--------|-----------|
| 00 | Setup India Boundaries | 0.1 | 1 | 📝 Copy code | `india_regions.geojson` |
| 01 | Preprocess Sample States | 1 | 2 | 📝 Copy code | `telangana/maharashtra_pop_*.npy` |
| 02 | Create HDF5 Dataset | 0.2 | 2-3 | 📝 Copy code | `india_sample.h5` (~500MB) |
| 03 | Clip Full India | 8-12 | 4 | ⏳ Overnight job | `india_pop_clipped_*.tif` + HDF5 |
| 04 | Model Architecture | 2 | 5 | 📝 Copy code | Tested model on sample |
| 05 | Progressive Training | 48+ | 5-8 | 🚀 Main training | `best_model.pt` |
| 06 | Inference & Predictions | 4 | 8-9 | 📝 Copy code | Prediction grids for 2025, 2030 |
| 07 | Gap Analysis | 6 | 9-10 | 📝 Copy code | Infrastructure recommendations |
| 08 | Dashboard & Deploy | 8 | 10-11 | 🌐 Streamlit app | Live interactive dashboard |

---

## 💾 DATA REQUIREMENTS

### Downloads Needed

| Source | Files | Size | Time | Week |
|--------|-------|------|------|------|
| **WorldPop** | `ind_ppp_2000-2020.tif` (5 files) | 3GB | 30-60min | 2 |
| **Natural Earth** | India boundaries | 50MB | 5min | 1 |

**Download Links:**
- WorldPop: https://data.worldpop.org/ (India, 2000/2005/2010/2015/2020)
- Natural Earth: https://naciscdn.org/naturalearth/10m/ (countries shapefile)

### Storage Structure

```
data/
├── raw/
│   ├── worldpop/
│   │   ├── ind_ppp_2000.tif  (600MB)
│   │   ├── ind_ppp_2005.tif  (600MB)
│   │   ├── ind_ppp_2010.tif  (600MB)
│   │   ├── ind_ppp_2015.tif  (600MB)
│   │   └── ind_ppp_2020.tif  (600MB)
│   └── india_regions.geojson (1MB)
├── processed/
│   ├── telangana_pop_*.npy  (20MB each)
│   ├── maharashtra_pop_*.npy  (25MB each)
│   ├── india_sample.h5  (500MB)
│   ├── india_population_full.h5  (~15GB, created in Notebook 03)
│   └── india_pop_clipped_*.tif  (2.5GB each, created in Notebook 03)
├── projections/
│   └── predictions_2030.tif  (created in Notebook 06)
└── viz/
    └── ...
```

---

## ⚡ EXECUTION TIMELINE

### WEEK 1: Foundation (16 hours)
- [ ] Day 1-2: Git + environment setup (2 hrs)
- [ ] Day 3-4: Implement core modules (6 hrs)
- [ ] Day 4-5: Test on existing Hyderabad data (2 hrs)
- [ ] Day 5: Notebook 00 - Setup boundaries (1 hr)
- [ ] Day 5: Download WorldPop data (5 hrs, background)

### WEEK 2-3: Sample Processing (20 hours)
- [ ] Day 1-2: Notebook 01 - Preprocess Telangana + Maharashtra (4 hrs)
- [ ] Day 2-3: Notebook 02 - Create HDF5 (1 hr)
- [ ] Day 3-4: Test HDF5 lazy loading (2 hrs)
- [ ] Day 4-5: Code review + optimization (2 hrs)
- [ ] Daily: Prepare for full India clipping (background)

### WEEK 4-5: Full-Scale Data (24 hours)
- [ ] Day 1-2: Notebook 03 - Clip all India (12 hrs, overnight)
- [ ] Day 2: Verify outputs (2 hrs)
- [ ] Day 3: Create full HDF5 (2 hrs)
- [ ] Day 4-5: Infrastructure code prep (8 hrs)

### WEEK 5-8: Model Training (48+ hours)
- [ ] Parallel: Notebook 04 - Model architecture (2 hrs)
- [ ] Continuous: Notebook 05 - Progressive training (48+ hrs)
  - Stage 1 (Coarse): 6 hours
  - Stage 2 (Medium): 15 hours
  - Stage 3 (Fine): 25 hours

### WEEK 8-10: Analysis & Results (16 hours)
- [ ] Day 1: Notebook 06 - Inference (4 hrs)
- [ ] Day 2-3: Notebook 07 - Gap analysis (6 hrs)
- [ ] Day 3-4: Results validation (4 hrs)
- [ ] Day 4-5: Report generation (2 hrs)

### WEEK 10-12: Deployment (20 hours)
- [ ] Day 1-3: Notebook 08 - Build dashboard (8 hrs)
- [ ] Day 3-4: Deploy to cloud or local server (8 hrs)
- [ ] Day 4-5: Testing + documentation (4 hrs)

### WEEK 12-16: Buffer (32 hours)
- [ ] Optimization + refinements
- [ ] Presentation preparation
- [ ] Final testing

---

## 🔧 DEVICE STRATEGY

### On Your Laptop (CPU)
```bash
# Set .env
CIVICPULSE_DEVICE=cpu
CIVICPULSE_BATCH_SIZE=4
CIVICPULSE_DATA_MODE=hdf5
CIVICPULSE_PATCH_SIZE=200
```
- **Good for**: Data prep, HDF5 creation, inference
- **Not for**: Training (too slow)
- **Memory mode**: HDF5 lazy loading (only 2-5GB active)

### On GPU PC
```bash
# Set .env
CIVICPULSE_DEVICE=cuda
CIVICPULSE_BATCH_SIZE=64
CIVICPULSE_DATA_MODE=numpy
CIVICPULSE_PATCH_SIZE=500
```
- **Good for**: Training (48 hrs → 4-6 hrs)
- **Good for**: Inference (1000x faster)
- **Memory mode**: Pre-load full array (40GB+ RAM)

### Hybrid Workflow
1. **Laptop** (Weeks 1-4): Download, preprocess, create HDF5
2. **Both in parallel** (Weeks 4-5): Laptop clips full India while GPU does sample training tests
3. **GPU** (Weeks 5-8): Full training on complete India dataset
4. **Laptop** (Weeks 8-10): Analysis, dashboard, deployment

---

## 🎯 ESTIMATED COSTS

### Compute Time
- Laptop CPU: ~120 hours (training) + 50 hours (preprocessing/analysis) = 170 hours ≈ **3 weeks continuous**
- GPU PC: ~15 hours (training) + 20 hours (inference/analysis) = 35 hours ≈ **1.5 days continuous**

### Storage
- **Raw data**: 3GB (WorldPop)
- **Processed data**: ~15GB (HDF5 + clipped TIFFs)
- **Model checkpoints**: 500MB
- **Total**: ~20GB

### Development Time
- **With this guide**: 12-16 weeks (feasible solo)
- **Without guidance**: 20-30 weeks (frequent rewrites)

---

## 🚨 CRITICAL GOTCHAS & FIXES

| Problem | Symptom | Solution |
|---------|---------|----------|
| **OutOfMemory on laptop** | Python crashes after 5 minutes | Use HDF5 chunks (256×256), never load full array |
| **Slow HDF5 reads** | Reading 1 chunk takes 10+ seconds | Adjust chunk size in Notebook 02, try (1, 512, 512) |
| **Git LFS not setup** | Files > 100MB fail to push | Do `git lfs install` before any commits, verify `.gitattributes` |
| **WorldPop files corrupt** | Rasterio throws errors | Re-download, verify SHA256 checksum |
| **Misaligned grids** | Model expects (H, W) but gets (W, H) | Always transpose to (height, width, time) in preprocessing |
| **Training doesn't converge** | Loss plateaus early | Check learning rate, reduce batch size, verify data normalization |
| **Predictions all zeros** | Model outputs blank grid | Verify test data is in same format as training, check normalization |

---

## ✅ SUCCESS MILESTONES

**By End of Week 1:**
- [x] Git repo ready with LFS + branches
- [x] Core modules implemented (region_manager, preprocessing, config)
- [x] Hyderabad test passes
- [x] WorldPop data downloaded

**By End of Week 3:**
- [x] Sample states preprocessed (Telangana + Maharashtra)
- [x] HDF5 file created and verified
- [x] Lazy loading confirmed working (<2GB active RAM)

**By End of Week 5:**
- [x] Full India clipped (overnight job completed)
- [x] Model architecture tested on sample
- [x] Training started with progress tracking

**By End of Week 8:**
- [x] Model converged (R² > 0.85 on test set)
- [x] 2025 and 2030 predictions generated
- [x] Results validated against known urban patterns

**By End of Week 10:**
- [x] Gap analysis complete
- [x] Infrastructure recommendations generated
- [x] Dashboard functional

**By End of Week 12:**
- [x] Deployed to cloud or local server
- [x] Final report generated
- [x] Presentation ready

---

## 📞 QUICK REFERENCE COMMANDS

```bash
# Check device auto-detection
python -c "import torch; print(f'Device: {torch.cuda.is_available()}')"

# Verify Git LFS
git lfs ls-files

# Count files
find data/raw -name "*.tif" | wc -l
find data/processed -name "*.npy" | wc -l

# Check HDF5 structure
python -c "import h5py; h5 = h5py.File('data/processed/india_sample.h5'); print(h5.keys())"

# Monitor training
tail -f logs/training.log

# Check disk usage
du -sh data/*/

# Commit progress
git add . && git commit -m "[LAPTOP] Notebook 03 complete - full India clipped"
```

---

## 📞 SUPPORT MATRIX

| Issue | Laptop Solution | GPU Solution |
|-------|-----------------|--------------|
| Out of RAM | Use HDF5 chunks, batch_size=2 | Pre-load array, batch_size=64 |
| Slow training | Skip to Stage 3, use LR schedule | Full training recommended |
| No GPU available | Use CPU with time buffer | SSH to GPU PC, run remotely |
| Predictions look wrong | Verify input normalization | Check train/test split data |

---

## 🎓 LEARNING RESOURCES

- **ConvLSTM Reference**: https://github.com/ndrplz/ConvLSTM_PyTorch
- **Population Forecasting**: https://www.nature.com/articles/s41467-022-30322-7
- **Rasterio Tutorial**: https://rasterio.readthedocs.io/
- **HDF5 Best Practices**: https://portal.hdfgroup.org/display/HDF5/HDF5+Best+Practices
- **FAISS GPU Search**: https://github.com/facebookresearch/faiss

---

## 🏁 FINAL CHECKLIST

Before declaring project COMPLETE:

- [ ] All 8 notebooks run end-to-end
- [ ] Model trained on full India dataset
- [ ] Predictions validated (R² > 0.85)
- [ ] Gap analysis identifies realistic locations
- [ ] Dashboard deploys without errors
- [ ] All code committed to Git with LFS
- [ ] README updated with results
- [ ] PDF report generated
- [ ] Presentation slides ready
- [ ] Share with stakeholders ✓

---

## 📧 NEXT IMMEDIATE ACTION

**Today (January 28):**
1. Open `CivicPulse_Production_Guide.md`
2. Read PHASE 0 (2 hours max)
3. Run Phase 0 commands in terminal
4. Verify all imports work
5. Commit initial setup to Git

**Tomorrow:**
1. Implement `src/region_manager.py`
2. Update other core modules
3. Run all tests
4. Start Notebook 00

**This Week:**
1. Complete Notebooks 00-02
2. Test on sample states
3. Prepare for full India clipping

---

**You've got everything. Let's build this!** 🚀

---

*Last Update: January 28, 2026*  
*Version: 2.0 All-India Ready*  
*Prepared by: AI Assistant*  
*For: Your CivicPulse India Project*