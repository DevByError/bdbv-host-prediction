# Quick Reference Card

## 🚀 Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (optional - models already included)
python scripts/run_training.py

# 3. Run web app
python app_advanced.py
```

Then visit: `http://localhost:5000`

---

## 🔮 Make Predictions

### Web API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"species":"Homo_sapiens"}'
```

### Python
```python
from src.pipelines.advanced_prediction_pipeline import AdvancedPredictionPipeline
pipeline = AdvancedPredictionPipeline(model_name="SVM")
result = pipeline.predict_for_species("Homo_sapiens")
print(f"{result['probability']:.1%}")  # 35.4%
```

---

## 📊 System Stats

| Metric | Value |
|--------|-------|
| Species | 41 in database |
| Records | 10,000 training |
| Features | 51 engineered |
| Best Model | SVM (F1: 0.462, AUC: 0.684) |
| Prediction Range | 16.89% - 38.34% |
| Avg Prediction | 28.83% |
| Database Confidence | 95% |

---

## 🔑 Critical Files

| File | Purpose |
|------|---------|
| `README.md` | **📖 READ THIS FIRST** - Full documentation |
| `app_advanced.py` | Web application |
| `scripts/run_training.py` | Model training |
| `results/models/SVM_model.pkl` | Trained model |
| `results/models/feature_engineer.pkl` | **⚠️ CRITICAL - Feature scaler** |
| `data/raw/filovirus_data_template.csv` | Dataset |

---

## 🎯 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/predict` | POST | Single species prediction |
| `/api/batch-predict` | POST | Multiple species |
| `/api/species/list` | GET | All species |
| `/api/species/search` | GET | Search species |
| `/api/species/<name>` | GET | Species details |
| `/api/database/stats` | GET | Database statistics |

---

## 💡 How It Works

### Known Species (In Database)
```
Input: "Homo_sapiens"
  ↓
Database lookup → Find exact features
  ↓
Feature scaling (using fitted scaler)
  ↓
SVM prediction
  ↓
Output: 35.4% susceptibility, 95% confidence
```

### Unknown Species (Not in Database)
```
Input: "Equus_zebra"
  ↓
Similarity search → Find 20 most similar species
  ↓
Interpolate features from neighbors
  ↓
Feature scaling
  ↓
SVM prediction
  ↓
Output: 34.7% susceptibility, 85% confidence
```

---

## ⚙️ Configuration

**Edit** `config/config.py` to change:
- Data paths
- Model directories
- Random seeds
- Train/test ratio

**Edit** `app_advanced.py` line 395 to change:
- Port (default 5000)
- Host (default 127.0.0.1)
- Debug mode

---

## 🧠 Features (51 Total)

### Viral (5 features)
GC content, CpG frequency, genome length, glycoprotein traits

### Host Biology (5 features)
Body mass, lifespan, geographic range

### Taxonomy (9 features)
Binary flags: primate, chiroptera, rodentia, carnivora, specific families

### Behavior/Ecology (8 features)
Diet type, social behavior, habitat type, human contact

### Risk Scores (19 features)
Composite risk indicators: bdbv_risk, contact_risk, taxonomic_risk, etc.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Port 5000 in use | Change port in app_advanced.py:395 |
| Slow predictions | First prediction slower (loading database) |
| Wrong predictions | Check `feature_engineer.pkl` exists |

---

## 📈 Model Training

**Input**: 10,000 records × 51 features  
**Process**: 4 models trained (SVM, XGBoost, RF, LR)  
**Output**: 
- Best model: SVM (saved to `SVM_model.pkl`)
- Feature scaler: Fitted `feature_engineer.pkl`
- Metrics: `model_performance.csv`

**Time**: ~2-5 minutes  
**When to retrain**: When adding new data or tuning hyperparameters

---

## 📁 Project Layout

```
Project Root
├── README.md (📖 START HERE)
├── app_advanced.py (Run this)
├── requirements.txt (pip install)
└── scripts/
    └── run_training.py (Training)
```

**Data**: `data/raw/filovirus_data_template.csv`  
**Models**: `results/models/*.pkl`  
**Config**: `config/config.py`  
**Code**: `src/` (organized by function)

---

## 🎓 Example Predictions

```
Homo_sapiens        → 35.4% (DB, 95% conf)   - High for primates
Gorilla_gorilla     → 38.3% (DB, 95% conf)   - Highest primate
Panthera_leo        → 20.1% (DB, 95% conf)   - Low for carnivores
Eidolon_helvum      → 29.8% (DB, 95% conf)   - Medium for bats
Equus_zebra         → 34.7% (Sim, 85% conf)  - Similar to E. asinus
```

---

## 🔐 Important Notes

1. **Feature Engineer is CRITICAL**
   - Contains fitted StandardScaler from training
   - Must exist for correct predictions
   - Never delete `results/models/feature_engineer.pkl`

2. **Predictions Vary by Species**
   - Before fix: All ~19.4% (bug)
   - After fix: 16.89% - 38.34% (correct)

3. **Confidence Levels**
   - Database species: 95% (direct lookup)
   - Unknown species: 51-85% (based on neighbor similarity)

4. **Model Performance**
   - SVM selected for best F1-Score (0.462)
   - Balanced performance across metrics
   - ROC-AUC: 0.684 (good discrimination)

---

## 📚 For More Details

- **Full Documentation**: See `README.md`
- **Cleanup Instructions**: See `CLEANUP_GUIDE.md`
- **Code**: Explore `src/` directory with inline comments
- **Configuration**: Edit `config/config.py`

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026-08-11  
**System Health**: All systems operational

