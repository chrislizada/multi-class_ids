# Multi-Class Intrusion Detection System (IDS)

Enhanced deep learning-based IDS for IoT networks using DAE-SMOTE and advanced classifiers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## Overview

This project implements a state-of-the-art intrusion detection system for IoT networks using a hybrid approach combining:
- **Denoising Autoencoders (DAE)** for robust feature extraction
- **Advanced SMOTE variants** for intelligent class balancing
- **Deep Learning classifiers** (1D-CNN, MLP, BiLSTM) with attention mechanisms
- **Ensemble learning** with optimized stacking

## Key Features

### Advanced Data Preprocessing
- Statistical feature engineering (std, variance, skewness, kurtosis)
- Automatic removal of constant and highly correlated features
- RobustScaler for outlier-resistant normalization
- One-hot encoding with unknown category handling
- **Consistent feature transformation** across train/test splits

### Denoising Autoencoder (DAE)
- Multiple architecture configurations (3-5 encoder layers)
- Adaptive noise injection strategies
- Latent dimension search (32, 64, 128, 256)
- Hyperparameter optimization with Optuna (20+ trials)
- Batch normalization and dropout regularization

### Advanced SMOTE Balancing
- **Borderline-SMOTE** for hard-to-classify boundary samples
- **ADASYN** for adaptive density-based synthesis
- **SMOTE-Tomek** for boundary cleaning
- **Auto-detection** of minority classes (no hardcoding)
- **Dynamic k_neighbors** adjustment for small classes
- Class-specific balancing strategies

### 1D-CNN Classifier
- Multi-scale kernel architecture (3x3, 5x5, 7x7)
- Parallel feature extraction paths
- Global max pooling aggregation
- **Focal Loss** for severe class imbalance
- Hyperparameter optimization (filters, kernels, dropout)

### MLP Classifier
- Deep architecture with residual skip connections
- Multiple activation functions (ReLU, SELU, PReLU)
- Layer-wise batch normalization
- Configurable depth (3-4 hidden layers)
- Focal loss integration

### BiLSTM with Attention
- Bidirectional LSTM for sequence context
- **Self-attention mechanism** for temporal focus
- Recurrent dropout for regularization
- Adaptive sequence reshaping
- Gradient clipping for stability

### Ensemble Model
- Weighted voting with **optimized weights** (Optuna)
- **XGBoost stacking** meta-learner
- Soft voting for probability calibration
- Individual model performance tracking

## Project Structure

```
multi-class_ids/
├── src/
│   ├── config.py                 # Configuration and hyperparameters
│   ├── preprocessing.py          # Data preprocessing module (FIXED)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dae.py               # Denoising Autoencoder
│   │   ├── smote_balancer.py    # SMOTE balancing (FIXED)
│   │   ├── cnn_classifier.py    # 1D-CNN classifier (FIXED)
│   │   ├── mlp_classifier.py    # MLP classifier (FIXED)
│   │   ├── lstm_classifier.py   # BiLSTM with attention (FIXED)
│   │   └── ensemble.py          # Ensemble model
│   └── utils/
│       ├── __init__.py
│       └── metrics.py           # Evaluation metrics and visualization
├── main.py                      # Main training pipeline (FIXED)
├── evaluate.py                  # Model evaluation script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Dataset directory
├── models/                      # Saved models directory
├── results/                     # Experiment results
└── logs/                        # Training logs
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM

### Setup

1. **Clone or download the project:**
```bash
cd "Multi-CLass_IDS"
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Workflow Overview

This implementation follows a complete end-to-end pipeline for multi-class intrusion detection:

```
1. Data Preparation (merge_dataset_sampled.py)
   └─> Download CIC IoT-DIAD 2024 dataset
   └─> Sample 10% from each attack class (~540K samples)
   └─> Merge 132 CSV files into single dataset
   └─> Output: data/merged_flow_dataset.csv

2. Data Preprocessing (src/preprocessing.py)
   └─> Handle missing values (median/mode imputation)
   └─> Feature engineering (std, variance, skewness, kurtosis)
   └─> Remove constant features (variance < threshold)
   └─> Remove highly correlated features (correlation > 0.95)
   └─> RobustScaler normalization
   └─> OneHotEncoder for categorical features
   └─> Train/test split (80/20)

3. Dimensionality Reduction (src/models/dae.py)
   └─> Denoising Autoencoder (DAE)
   └─> Optuna hyperparameter optimization (20 trials)
   └─> Reduce features: 84+ → 32-256 latent dimensions
   └─> Save encoder for inference

4. Class Balancing (src/models/smote_balancer.py)
   └─> Calculate class weights (before SMOTE)
   └─> Auto-detect minority classes
   └─> Apply Borderline-SMOTE / ADASYN
   └─> Dynamic k_neighbors adjustment
   └─> Output: balanced training set

5. Model Training (main.py)
   ├─> MLP Classifier (src/models/mlp_classifier.py)
   │   └─> Optuna optimization: layers, neurons, activation, dropout
   │   └─> Focal Loss for class imbalance
   │   └─> Early stopping + best model checkpoint
   │
   ├─> 1D-CNN Classifier (src/models/cnn_classifier.py)
   │   └─> Optuna optimization: filters, kernels, dropout
   │   └─> Multi-kernel architecture (3x3, 5x5, 7x7)
   │   └─> Focal Loss + class weights
   │
   ├─> BiLSTM Classifier (src/models/lstm_classifier.py)
   │   └─> Optuna optimization: LSTM units, layers, dropout
   │   └─> Self-attention mechanism
   │   └─> Bidirectional processing
   │
   └─> Ensemble Model (src/models/ensemble.py)
       └─> Weighted voting (optimized weights)
       └─> XGBoost stacking meta-learner
       └─> Soft voting for probability calibration

6. Evaluation (src/utils/metrics.py)
   └─> Generate confusion matrices
   └─> ROC curves (One-vs-Rest)
   └─> Classification reports (per-class metrics)
   └─> Save all metrics to JSON
   └─> Create visualizations (PNG)

7. Results (results/experiment_YYYYMMDD_HHMMSS/)
   └─> summary_results.csv - Model comparison
   └─> Per-model directories with metrics and trained models
   └─> Confusion matrices, ROC curves, reports
```

## Usage

### Step 1: Download and Merge Dataset

```bash
# Download CIC IoT-DIAD 2024 dataset
# Extract to: data/ciciot_idad_2024/

# Merge with 10% sampling (recommended)
python merge_dataset_sampled.py

# Custom sampling (e.g., 20%)
python merge_dataset_sampled.py --sample-fraction 0.2

# Specify custom paths
python merge_dataset_sampled.py \
  --dataset-dir data/ciciot_idad_2024 \
  --output data/merged_flow_dataset.csv
```

### Step 2: Train Models

**Option A: Full Pipeline (Recommended)**
```bash
# Train all models with hyperparameter optimization
python main.py --data data/merged_flow_dataset.csv

# What happens:
# - Preprocessing (feature engineering, normalization)
# - DAE dimensionality reduction (optimized)
# - SMOTE class balancing (optimized)
# - Train MLP, CNN, LSTM, Ensemble (each optimized)
# - Generate evaluation metrics and visualizations
# - Save all models and results
```

**Option B: Fast Training (Skip Optimization)**
```bash
# Use default hyperparameters (faster)
python main.py --data data/merged_flow_dataset.csv --no-optimize

# Time: ~30-60 minutes (vs 2-4 hours with optimization)
```

**Option C: Custom Configuration**
```bash
# Without DAE dimensionality reduction
python main.py --data data/merged_flow_dataset.csv --no-dae

# Without SMOTE balancing
python main.py --data data/merged_flow_dataset.csv --no-smote

# Minimal (fastest, for testing)
python main.py --data data/merged_flow_dataset.csv --no-optimize --no-dae --no-smote
```

### Step 3: Evaluate Trained Models

```bash
# Evaluate on new test data
python evaluate.py \
  --experiment results/experiment_20241207_120000 \
  --data data/test_set.csv

# Specify custom output directory
python evaluate.py \
  --experiment results/experiment_20241207_120000 \
  --data data/test_set.csv \
  --output results/evaluation_test
```

### Step 4: View Results

```bash
# View summary results
cat results/experiment_20241207_120000/summary_results.csv

# View per-model metrics
cat results/experiment_20241207_120000/cnn/report.txt
cat results/experiment_20241207_120000/mlp/report.txt
cat results/experiment_20241207_120000/lstm/report.txt
cat results/experiment_20241207_120000/ensemble/report.txt

# Visualizations (confusion matrices, ROC curves)
# Located in: results/experiment_20241207_120000/{model_name}/*.png
```

### Dataset Requirements

This project is designed for the **CIC IoT-DIAD 2024** dataset.

**Format**: CSV or Parquet

**Required column**: `label` or `Label` (containing attack class names)

**Supported classes** (automatically detected):
- Benign
- BruteForce
- DDoS
- DoS
- Mirai
- Recon
- Spoofing
- Web-Based

**Features**: Numerical and/or categorical (handled automatically)

**Missing values**: Handled automatically (median for numerical, mode for categorical)

**Download**: See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed instructions on obtaining and preparing the CIC IoT-DIAD 2024 dataset.

**Dataset Sampling**: Due to the large size of the CIC IoT-DIAD 2024 dataset (~5-6M samples), this implementation uses **10% stratified sampling** per attack class to balance computational efficiency with model performance. The `merge_dataset_sampled.py` script samples each CSV file individually before merging to avoid memory issues.

**Merging the dataset**:
```bash
# Default: 10% sampling per attack class
python merge_dataset_sampled.py

# Custom sampling (e.g., 20%)
python merge_dataset_sampled.py --sample-fraction 0.2

# Specify custom paths
python merge_dataset_sampled.py --dataset-dir data/ciciot_idad_2024 --output data/merged_flow_dataset.csv
```

**Example training usage**:
```bash
# Using merged flow-based features (recommended)
python main.py --data data/merged_flow_dataset.csv

# Quick test without optimization
python main.py --data data/merged_flow_dataset.csv --no-optimize
```

## Output Structure

Each experiment creates a timestamped directory: `results/experiment_YYYYMMDD_HHMMSS/`

### Per-Model Results (CNN, MLP, LSTM, Ensemble)

```
results/experiment_YYYYMMDD_HHMMSS/
├── cnn/
│   ├── confusion_matrix.png           # Confusion matrix heatmap
│   ├── confusion_matrix_normalized.png # Normalized confusion matrix
│   ├── classification_report.png       # Per-class precision/recall/F1
│   ├── training_history.png           # Loss and accuracy curves
│   ├── roc_curves.png                 # ROC curves (all classes)
│   ├── metrics.json                   # All metrics in JSON
│   ├── report.txt                     # Human-readable summary
│   ├── cnn_model.h5                   # Trained model
│   ├── best_params.pkl                # Optimized hyperparameters
│   └── history.pkl                    # Training history
├── mlp/
│   └── ... (same structure)
├── lstm/
│   └── ... (same structure)
├── ensemble/
│   └── ... (same structure)
├── dae/
│   ├── encoder.h5                     # Trained encoder
│   ├── decoder.h5                     # Trained decoder
│   ├── autoencoder.h5                 # Full autoencoder
│   └── best_params.pkl                # Optimized DAE params
├── preprocessor.pkl                    # Fitted preprocessor
├── smote_balancer.pkl                  # SMOTE configuration
└── summary_results.csv                 # Model comparison table
```

## Configuration

Edit `src/config.py` to customize:

### Hyperparameter Search Spaces
```python
CNN_CONFIG = {
    'filters': [[64, 128, 256], [128, 256, 512]],
    'kernel_sizes': [[3, 5, 7], [5, 7, 9]],
    'dropout_rate': [0.3, 0.4, 0.5],
    ...
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    'use_focal_loss': True,        # Focal loss for class imbalance
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'use_class_weights': True,
    'use_early_stopping': True,
    ...
}
```

### SMOTE Configuration
```python
SMOTE_CONFIG = {
    'method': ['borderline', 'adasyn', 'smote'],
    'k_neighbors': [3, 5, 7],
    'sampling_strategy': 'not majority'
}
```

## Dataset Preparation

### Merging CIC IoT-DIAD 2024 Dataset

The dataset contains 132 CSV files across 8 attack categories with ~5-6 million samples total. Three merge scripts are provided:

**1. Recommended: Sample-Limited Merge (5% sampling)**
```bash
python merge_dataset_sampled.py --sample-fraction 0.05
```
- Samples 5% from each CSV file individually (memory-efficient)
- Output: ~1.39M samples total
- Prevents out-of-memory errors on limited hardware
- Maintains class distribution
- Faster training times

**2. Memory-Efficient Merge (Chunked Processing)**
```bash
python merge_dataset_efficient.py
```
- Processes files in 50K-row chunks
- Output: Full dataset (~5-6M samples)
- Requires sufficient disk space

**3. Standard Merge (Full Dataset)**
```bash
python merge_dataset.py
```
- Loads all data into memory at once
- Requires 16GB+ RAM
- Output: Full dataset (~5-6M samples)

**Adjusting sample fraction:**
```bash
# 5% sampling (~1.39M samples) - Recommended for faster training
python merge_dataset_sampled.py --sample-fraction 0.05

# 10% sampling (~2.78M samples) - More data, longer training
python merge_dataset_sampled.py --sample-fraction 0.10

# 20% sampling (~5.56M samples) - Requires significant resources
python merge_dataset_sampled.py --sample-fraction 0.2
```

## Performance Metrics

All models are evaluated using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification correctness |
| **Precision** (Macro) | Average precision across all classes |
| **Precision** (Weighted) | Class-size weighted precision |
| **Recall** (Macro) | Average recall across all classes |
| **Recall** (Weighted) | Class-size weighted recall |
| **F1-Score** (Macro) | Harmonic mean of precision/recall |
| **F1-Score** (Weighted) | Class-size weighted F1 |
| **ROC AUC** | Area under ROC curve (multi-class OvR) |
| **Confusion Matrix** | Detailed per-class predictions |

## Expected Performance

Performance metrics will be generated after training on the CIC IoT-DIAD 2024 dataset.

### Dataset Configuration
- **Sampling**: 5% stratified sampling per attack class (configurable via `--sample-fraction`)
- **Total Samples**: ~1.39M samples after 5% sampling
- **Attack Classes**: 8 categories with natural class imbalance
  - DoS: ~1,157,191 samples (83.22%) - Highly dominant
  - DDoS: ~173,930 samples (12.51%)
  - Recon: ~22,108 samples (1.59%)
  - Benign: ~19,916 samples (1.43%)
  - Mirai: ~8,723 samples (0.63%)
  - Spoofing: ~7,861 samples (0.57%)
  - Web-Based: ~566 samples (0.04%) - Extremely rare
  - BruteForce: ~181 samples (0.01%) - Extremely rare
- **Features**: 62 features after preprocessing
  - Original: 84 flow-based features
  - Dropped: 6 identifier columns (Flow ID, IPs, Ports, Timestamp)
  - Added: 27 statistical features (std, variance, CV, log transforms)
  - Removed: 10 constant features + 32 highly correlated features
  - Final: 62 numerical features
- **Class Balancing**: Borderline-SMOTE / ADASYN applied to minority classes (BruteForce, Web-Based, Spoofing, Mirai, Benign, Recon)
- **Optimization Trials**: 5 trials per model (reduced from 50 for faster training)

### Models to be Evaluated

| Model | Description | Key Features |
|-------|-------------|--------------|
| **DAE** | Denoising Autoencoder | Dimensionality reduction: 62 → 32 features |
| **MLP** | Multi-Layer Perceptron | Residual connections, batch normalization, 3-4 hidden layers |
| **1D-CNN** | Convolutional Neural Network | Multi-kernel (3x3, 5x5, 7x7), parallel feature extraction |
| **BiLSTM+Attention** | Bidirectional LSTM | Self-attention mechanism, recurrent dropout |
| **Ensemble** | Stacking ensemble | XGBoost meta-learner combining CNN+MLP+LSTM predictions |

### Optimized Hyperparameters (5% Dataset, 5 Trials)

**Denoising Autoencoder (DAE)**
- Latent dimensions: 32
- Encoder layers: [1024, 512, 256]
- Noise factor: 0.2
- Dropout rate: 0.2
- Learning rate: 0.001
- Batch size: 128
- Validation loss: 1.12e+12

**Note:** CNN, MLP, LSTM, and Ensemble hyperparameters will be updated after training completes.

### Evaluation Metrics

All experiments will report:
- **Overall Accuracy** - Correct classifications across all classes
- **Precision** (Macro/Weighted) - Per-class and weighted average
- **Recall** (Macro/Weighted) - Per-class and weighted average  
- **F1-Score** (Macro/Weighted) - Harmonic mean of precision/recall
- **ROC AUC** - Area under ROC curve (One-vs-Rest)
- **Confusion Matrix** - Detailed per-class performance
- **Per-Class Metrics** - Individual F1, precision, recall for each attack type

### Training Time Estimates

**Hardware**: EC2 g4dn.2xlarge (Tesla T4 GPU, 5% sampling, 5 trials)
- **DAE Optimization**: ~1-1.5 hours (5 trials)
- **SMOTE Balancing**: ~5-10 minutes
- **CNN Training**: ~15-20 minutes (5 trials)
- **MLP Training**: ~15-20 minutes (5 trials)
- **LSTM Training**: ~20-30 minutes (5 trials)
- **Ensemble Training**: ~5-10 minutes
- **Total Time**: ~2.5-3 hours for full pipeline with optimization
- **Cost**: ~$1.90-2.25 (on-demand), ~$0.55-0.90 (spot)

**With CPU only** (no GPU): ~8-12 hours

**Fast Mode** (--no-optimize): ~30-45 minutes total

### Output Location

Results are saved in `results/experiment_YYYYMMDD_HHMMSS/`:
- `summary_results.csv` - Model comparison table
- `cnn/`, `mlp/`, `lstm/`, `ensemble/` - Per-model metrics and visualizations
- Confusion matrices, ROC curves, classification reports
- Train/validation/test split: 60% / 20% / 20%

## Bug Fixes & Improvements

### Critical Bugs Fixed (v1.1)
**OneHotEncoder initialization** - Prevented errors during test data processing  
**Feature dimension mismatch** - Ensured consistent features across train/test  
**Hardcoded minority classes** - Auto-detection based on distribution  
**Memory leaks in optimization** - Added proper Keras session cleanup  
**SMOTE k_neighbors validation** - Dynamic adjustment for small classes  
**Class weight timing** - Calculated before SMOTE for accurate weights  

See `BUGS_FOUND.md` and `BUGFIXES_APPLIED.md` for detailed information.

## Key Technical Improvements

| Improvement | Description |
|-------------|-------------|
| **Focal Loss** | Addresses class imbalance by down-weighting easy examples (α=0.25, γ=2.0) |
| **Borderline-SMOTE** | Generates synthetic samples near decision boundaries for hard-to-classify minority classes |
| **ADASYN** | Adaptive density-based sampling that focuses on difficult regions |
| **Multi-kernel 1D-CNN** | Parallel convolution paths (3x3, 5x5, 7x7) for multi-scale feature extraction |
| **BiLSTM+Attention** | Bidirectional LSTM with self-attention for temporal context and focus |
| **MLP with Residual Connections** | Skip connections to prevent gradient vanishing in deep networks |
| **Ensemble Stacking** | XGBoost meta-learner combines predictions from CNN, MLP, and LSTM |
| **Optuna Optimization** | Bayesian hyperparameter optimization with 20+ trials per model |
| **Advanced Feature Engineering** | Statistical features (std, variance, skewness, kurtosis) added to flow features |
| **RobustScaler** | Outlier-resistant normalization using median and IQR |
| **Auto-detection of Minority Classes** | Dynamic identification based on distribution, no hardcoding |

## Testing & Validation

The implementation has been validated for:
- Small datasets (< 100 samples per class)
- Highly imbalanced datasets (1:100 ratio)
- Mixed categorical/numerical features
- Missing values handling
- Consistent train/test preprocessing
- Memory efficiency during optimization
- Cross-platform compatibility (Windows/Linux/Mac)

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{lizada2024multiclass,
  title={Multi-Class Intrusion Detection in IoT Networks Using DAE-SMOTE and Deep Learning Classifiers},
  author={Lizada, Christopher},
  year={2024},
  school={De La Salle University},
  address={Manila, Philippines}
}
```

## License

This project is for academic and research purposes only.

## Contributing

Found a bug or have a suggestion? Please:
1. Check `BUGS_FOUND.md` for known issues
2. Open an issue with detailed description
3. Include dataset characteristics and error logs

## Contact

**Author**: Christopher Lizada  
**Email**: christopher_lizada@dlsu.edu.ph  
**Institution**: De La Salle University, Manila, Philippines

## Acknowledgments

- Canadian Institute for Cybersecurity for the CIC IoT-DIAD 2024 dataset
- De La Salle University Department of Electronics and Computer Engineering
- TensorFlow and Scikit-learn development teams

---

**Last Updated**: December 2024  
**Version**: 1.1 (Bug Fixes Applied)
