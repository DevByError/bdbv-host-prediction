# Documentation Index

Choose which document to read based on your need:

---

## 🎯 By Use Case

### "I want to use the system NOW"
→ **Read**: `QUICK_REFERENCE.md`  
- 3-step quick start
- API endpoints
- Example predictions
- Troubleshooting

**Time to read**: 5 minutes

---

### "I need complete system documentation"
→ **Read**: `README.md`  
- System overview
- Full API reference
- Feature descriptions
- Model performance details
- Usage examples
- Configuration options

**Time to read**: 20 minutes

---

### "I want to clean up the project"
→ **Read**: `CLEANUP_GUIDE.md`  
- Which files to remove
- Final project structure
- Verification checklist
- Git configuration

**Time to read**: 10 minutes

---

### "I need to understand the bug fix"
→ **Read**: Source code comments in:
- `src/pipelines/advanced_prediction_pipeline.py` (lines 47-52, 71-88, 289-302)
- `scripts/run_training.py` (lines 185-193)

**Key points**:
- Lines 71-88: `_load_feature_engineer()` method loads saved scaler
- Line 291: `transform()` instead of `fit_transform()` for correct scaling
- Line 297-298: Drop target column before prediction
- Lines 188-193: Save fitted feature engineer during training

**Time to read**: 10 minutes

---

## 📖 Document Overview

### QUICK_REFERENCE.md
**Purpose**: Get started immediately  
**Contents**:
- 3-step quick start
- API examples
- System statistics
- Critical files list
- Troubleshooting

**Best for**: Quick lookup, getting started, cheat sheet

---

### README.md
**Purpose**: Complete system documentation  
**Contents**:
- System overview & features
- Quick start (expanded)
- Project structure
- All API endpoints
- Model performance
- Feature descriptions
- Prediction methods
- Implementation details
- Usage examples
- Configuration
- Troubleshooting
- Dataset information
- Training details

**Best for**: Understanding the full system, deep dive

---

### CLEANUP_GUIDE.md
**Purpose**: Prepare project for production/deployment  
**Contents**:
- Files to remove (temporary/debug)
- Python cache cleanup
- Final production structure
- Critical files to keep
- Verification checklist
- Git configuration

**Best for**: Organizing the codebase, before deployment

---

### DOCUMENTATION_INDEX.md
**Purpose**: This file - help finding the right documentation  
**Contents**:
- Navigation by use case
- Document overview
- Reading time estimates
- Code location references

**Best for**: Finding the right documentation

---

## 🔍 By Topic

### Getting Started
1. `QUICK_REFERENCE.md` (5 min)
2. `README.md` - "Quick Start" section (5 min)

### Making Predictions
- `README.md` - "Usage Examples" section
- `QUICK_REFERENCE.md` - "Make Predictions" section

### Understanding the Data
- `README.md` - "Data Features" section
- `README.md` - "Dataset Information" section

### Deploying to Production
1. `CLEANUP_GUIDE.md` - Remove temporary files
2. `README.md` - "Configuration" section
3. `README.md` - "Troubleshooting" section

### Understanding the Bug Fix
- `src/pipelines/advanced_prediction_pipeline.py` - Code comments
- `README.md` - "Critical Implementation Details" section

### Model Performance
- `QUICK_REFERENCE.md` - "System Stats" table
- `README.md` - "Model Performance" section
- `results/model_performance.csv` - CSV metrics

### API Reference
- `README.md` - "API Endpoints" section
- `QUICK_REFERENCE.md` - "API Endpoints" table

### Configuration
- `README.md` - "Configuration" section
- `config/config.py` - Actual config file
- `QUICK_REFERENCE.md` - "Configuration" section

---

## 📋 Quick Links to Key Sections

| Need | Document | Section |
|------|----------|---------|
| Installation | README.md | Quick Start |
| API usage | QUICK_REFERENCE.md | Make Predictions |
| Endpoints | README.md | API Endpoints |
| Features | README.md | Data Features |
| Model info | QUICK_REFERENCE.md | System Stats |
| Troubleshoot | README.md | Troubleshooting |
| Configuration | README.md | Configuration |
| Cleanup | CLEANUP_GUIDE.md | Files to Remove |

---

## 🎓 Recommended Reading Order

### For New Users
1. `QUICK_REFERENCE.md` (5 min) - Get oriented
2. `README.md` "Quick Start" (5 min) - Run it
3. `README.md` "Usage Examples" (5 min) - Try it
4. `README.md` full read (15 min) - Understand it

**Total**: ~30 minutes to full understanding

### For Developers
1. `README.md` - Full read
2. `src/pipelines/advanced_prediction_pipeline.py` - Code walkthrough
3. `src/database/species_database.py` - Database logic
4. `scripts/run_training.py` - Training pipeline

**Total**: 1-2 hours for complete code understanding

### For DevOps/Deployment
1. `QUICK_REFERENCE.md` - System overview
2. `README.md` "Configuration" - Setup options
3. `CLEANUP_GUIDE.md` - Prepare for production
4. `README.md` "Troubleshooting" - Know the issues

**Total**: 30-45 minutes for deployment readiness

---

## 📄 All Documentation Files

| File | Type | Audience | Length |
|------|------|----------|--------|
| `README.md` | Comprehensive | Everyone | ~300 lines |
| `QUICK_REFERENCE.md` | Cheat sheet | Quick lookup | ~200 lines |
| `CLEANUP_GUIDE.md` | How-to | DevOps/Cleanup | ~150 lines |
| `DOCUMENTATION_INDEX.md` | Navigation | Everyone | This file |
| `requirements.txt` | Config | Developers | ~15 lines |
| `config/config.py` | Config | Developers | Settings |
| Source code | Code | Developers | ~5000 lines |

---

## 🔄 Update Frequency

These documents are updated:
- **README.md**: After major feature changes
- **QUICK_REFERENCE.md**: After API changes
- **CLEANUP_GUIDE.md**: Rarely (only if project structure changes)
- **DOCUMENTATION_INDEX.md**: Whenever documents are added/removed

**Last updated**: 2026-08-11

---

## 💬 Finding Information

### "How do I...?"
→ Check `README.md` "Usage Examples" section

### "What does this...?"
→ Check `README.md` "Data Features" section or source code comments

### "Why is...?"
→ Check `README.md` "Critical Implementation Details" section

### "How do I fix...?"
→ Check `README.md` "Troubleshooting" section

### "What files should I delete?"
→ Check `CLEANUP_GUIDE.md`

### "What are the API endpoints?"
→ Check `QUICK_REFERENCE.md` or `README.md`

### "How do I deploy?"
→ Check `CLEANUP_GUIDE.md` then `README.md` configuration

---

## 🎯 One-Minute Summary

**BDBV Host Prediction System**:
- Predicts animal susceptibility to Filovirus
- 10,000 training records, 41 species
- 51 engineered features
- SVM model (68.4% AUC)
- Flask API for predictions
- Ready for production use

**To use**:
1. `pip install -r requirements.txt`
2. `python app_advanced.py`
3. POST to `/api/predict` with species name

**To understand**:
- Read `README.md` for comprehensive documentation
- Read `QUICK_REFERENCE.md` for quick lookup

---

**Status**: ✅ Fully Documented  
**Ready for**: Production use, deployment, development  
**Questions?**: Check the appropriate documentation above

