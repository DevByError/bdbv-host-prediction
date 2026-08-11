# BDBV Host Susceptibility Prediction

## 🎯 Project Overview

Machine learning framework for predicting **Bundibugyo ebolavirus (BDBV)** host susceptibility using a pan-filovirus approach.

### Key Features

- **Multi-model Ensemble**: Random Forest, XGBoost, Logistic Regression, SVM, GNN
- **Biological Feature Engineering**: Codon usage analysis, NPC1 binding prediction, phylogenetic features
- **Explainable AI**: SHAP, LIME, feature importance with biological context
- **Publication-Ready**: High-DPI vector graphics for journals
- **Production-Grade**: Modular architecture, comprehensive testing, Docker support

### Research Question

> Can machine learning models identify host species potentially susceptible to Bundibugyo ebolavirus by integrating multi-omic viral features with host ecological characteristics?

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-org/bdbv-host-prediction.git
cd bdbv-host-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .



🚀 Quick Start
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline

# Train model
pipeline = TrainingPipeline()
results = pipeline.run()

# Predict on new hosts
predictor = PredictionPipeline()
predictions = predictor.predict_new_hosts("data/new_hosts.csv")




📊 Running the Pipeline
# Train and evaluate
python scripts/run_training.py

# Generate predictions
python scripts/run_prediction.py --input data/new_hosts.csv --output results/predictions.csv

# Validate results
python scripts/run_validation.py --predictions results/predictions.csv

# Generate all figures
python scripts/generate_figures.py




📁 Project Structure

bdbv_host_prediction/
├── config/          # Configuration
├── data/            # Data files (raw, processed, external)
├── src/             # Source code
│   ├── data/        # Data loading & validation
│   ├── features/    # Feature engineering
│   ├── models/      # ML models (classical, GNN, ensemble)
│   ├── evaluation/  # Metrics & cross-validation
│   ├── explainability/ # SHAP, LIME
│   ├── pipelines/   # Training, prediction, deployment
│   ├── utils/       # Logging, serialization, timing
│   └── visualization/ # Publication figures, maps, dashboard
├── scripts/         # Run scripts
├── notebooks/       # Jupyter notebooks
├── tests/           # Unit tests
├── results/         # Outputs (figures, models, reports)
└── logs/            # Log files


🧪 Testing

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models/test_models.py -v



📈 Outputs
All results are saved to the results/ directory:

Figures: 8 publication-quality figures (PNG, SVG)

Models: Best model saved as pickle file

Reports: Performance metrics, metadata, predictions



🔬 Citation

@article{bdbv_host_prediction_2024,
  title={Predicting Potential Reservoir Hosts of Bundibugyo Ebolavirus: 
         A Pan-Filovirus Machine Learning Approach},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}

