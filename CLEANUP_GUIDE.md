# Project Cleanup Guide

## Files to Remove (Temporary/Debug)

These files were used during development/debugging and are not needed in production:

```
test_predictions_fixed.py       # Debug test script
test_system_complete.py         # Comprehensive test script  
debug_features.py               # Feature debugging script
FIX_SUMMARY.md                  # Fix documentation (superseded by README)
SYSTEM_STATUS.md                # Status report (superseded by README)
```

**Delete these files:**
```bash
rm test_predictions_fixed.py
rm test_system_complete.py
rm debug_features.py
rm FIX_SUMMARY.md
rm SYSTEM_STATUS.md
```

## Python Cache to Remove

Also remove Python cache files:
```bash
# Remove all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} +

# Remove all .pyc files
find . -name "*.pyc" -delete

# Remove .pytest_cache
rm -rf .pytest_cache
```

## Final Production Structure

After cleanup, your project should look like:

```
bdbv_host_prediction/
├── README.md                          # ✅ KEEP - Complete documentation
├── CLEANUP_GUIDE.md                   # ✅ KEEP - This file
├── requirements.txt                   # ✅ KEEP - Dependencies
├── app_advanced.py                    # ✅ KEEP - Flask app
│
├── scripts/
│   └── run_training.py               # ✅ KEEP - Training pipeline
│
├── src/                               # ✅ KEEP - All source code
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── database/
│   ├── evaluation/
│   ├── utils/
│   └── explainability/
│
├── config/                            # ✅ KEEP - Configuration
│
├── templates/                         # ✅ KEEP - Web UI
│   └── index_advanced.html
│
├── data/                              # ✅ KEEP - Dataset
│   └── raw/
│       └── filovirus_data_template.csv
│
└── results/                           # ✅ KEEP - Models & metrics
    ├── models/
    │   ├── SVM_model.pkl
    │   ├── feature_engineer.pkl       # 🔴 CRITICAL
    │   └── ...
    └── model_performance.csv
```

## Important Notes

### Files to NEVER Delete
- ❌ `results/models/feature_engineer.pkl` - **CRITICAL FOR PREDICTIONS**
- ❌ `results/models/SVM_model.pkl` - **Best model**
- ❌ `src/pipelines/advanced_prediction_pipeline.py` - **Core prediction engine**
- ❌ `src/database/species_database.py` - **Species database & similarity matching**
- ❌ `data/raw/filovirus_data_template.csv` - **Training dataset**

### What Was Documented
All important information has been consolidated into:
- **README.md** - Complete system documentation
  - Quick start guide
  - API endpoints
  - Model performance
  - Feature descriptions
  - Troubleshooting
  - Usage examples

Previous documentation files (FIX_SUMMARY.md, SYSTEM_STATUS.md) are now superseded by README.md.

## Verification Checklist

After cleanup, verify everything still works:

```bash
# 1. Check app starts
python app_advanced.py &

# 2. Test API (in another terminal)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"species":"Homo_sapiens"}'

# 3. Expected response: Probability ~35.4%, Confidence 95%

# 4. Kill the app
killall python
```

## Quick Reference: What Each Directory Does

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code (models, pipelines, features) |
| `scripts/` | Training and utility scripts |
| `config/` | Configuration and logging setup |
| `templates/` | Web UI HTML (for Flask app) |
| `data/` | Raw dataset (CSV file) |
| `results/` | Trained models and performance metrics |

## Storage Space

**Typical sizes after training:**
- Models: ~25 MB (4 model files + feature engineer)
- Data: ~2 MB (10k records CSV)
- Source code: ~1 MB
- **Total: ~30 MB** (very efficient)

## Git Configuration (if using version control)

Add to `.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
results/models/*.pkl
.env
venv/
```

This prevents accidentally committing large pickle files.

---

**After cleanup, your project will be clean, production-ready, and well-documented!** ✅
