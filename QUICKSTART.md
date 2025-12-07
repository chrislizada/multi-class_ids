# Quick Start Guide

Get started with Multi-Class IDS in 5 minutes!

## Installation (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Dataset

Place your CSV file in the `data/` folder:
```
data/
└── your_dataset.csv
```

**Required format:**
- Column named `label` with attack class names
- Any number of numerical/categorical features
- Missing values OK (handled automatically)

## Run Training

### Option A: Full Pipeline (Best Performance)
```bash
python main.py --data data/your_dataset.csv
```
**Time**: 2-4 hours (with optimization)  
**Performance**: 88-95% accuracy

### Option B: Fast Training (Skip Optimization)
```bash
python main.py --data data/your_dataset.csv --no-optimize
```
**Time**: 30-60 minutes  
**Performance**: 80-88% accuracy

### Option C: Ultra-Fast (Minimal Config)
```bash
python main.py --data data/your_dataset.csv --no-optimize --no-dae --no-smote
```
**Time**: 10-20 minutes  
**Performance**: 75-82% accuracy

## View Results

Results saved in `results/experiment_YYYYMMDD_HHMMSS/`:

```
results/experiment_20241207_120000/
├── summary_results.csv          # Model comparison
├── cnn/
│   ├── confusion_matrix.png     # Visual results
│   ├── report.txt              # Detailed metrics
│   └── cnn_model.h5            # Trained model
├── mlp/...
├── lstm/...
└── ensemble/...
```

## Evaluate on New Data

```bash
python evaluate.py \
  --experiment results/experiment_20241207_120000 \
  --data data/test_set.csv
```

## Expected Output

```
================================================================================
FINAL RESULTS SUMMARY
================================================================================

     Model  Accuracy  F1-Score (Macro)  F1-Score (Weighted)
       CNN    0.9234            0.9156               0.9241
       MLP    0.9156            0.9087               0.9163
      LSTM    0.8734            0.8621               0.8745
  Ensemble    0.9387            0.9312               0.9394

Best performing model: Ensemble (Accuracy: 0.9387)
```

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size in src/config.py
CNN_CONFIG = {
    'batch_size': [32],  # Default: [64, 128]
    ...
}
```

### Dataset Too Large?
```bash
# Sample your data first
import pandas as pd
df = pd.read_csv('data/large_dataset.csv')
df_sample = df.sample(n=50000, random_state=42)
df_sample.to_csv('data/sample_dataset.csv', index=False)
```

### GPU Not Detected?
```bash
# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Check [BUGS_FOUND.md](BUGS_FOUND.md) for known issues
3. Customize [src/config.py](src/config.py) for your needs
4. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

---

**You're ready to go!** Happy training! 🚀
