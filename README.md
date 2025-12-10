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

This project uses the **CIC IoT-DIAD 2024 Packet-Based Dataset**.

**Dataset Type**: Packet-Based Features (RECOMMENDED)

**Why Packet-Based over Flow-Based:**
- ✅ **Richer features:** 100+ packet-level statistics vs 84 flow-level aggregates
- ✅ **Better class separation:** Attack signatures more visible at packet level
- ✅ **More attack types:** 33+ granular attack classifications
- ✅ **Better performance:** Expected 40-65% Macro F1 vs 11% with flow-based
- ✅ **Detailed analysis:** Per-packet timing, header fields, protocol flags

**Format**: CSV

**Required column**: `label` or `Label` (containing attack class names)

**Supported classes** (33+ attack types automatically detected):
- **Benign:** BenignTraffic
- **DDoS Variants:** DDoS-ACK_Fragmentation, DDoS-HTTP_Flood, DDoS-ICMP_Fragmentation, DDoS-SlowLoris, DDoS-SynonymousIP_Flood, DDoS-TCP_Flood, DDoS-UDP_Flood, DDoS-UDP_Fragmentation
- **DoS Variants:** DoS-HTTP_Flood, DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood
- **Reconnaissance:** Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan
- **Web Attacks:** BrowserHijacking, CommandInjection, SqlInjection, XSS, VulnerabilityScan, Uploading_Attack
- **IoT Malware:** Mirai-greip_flood, Backdoor_Malware
- **Network Attacks:** DNS_Spoofing, MITM-ArpSpoofing, DictionaryBruteForce

**Features**: 100+ packet-level numerical features (handled automatically)

**Missing values**: Handled automatically (median for numerical, mode for categorical)

**Download**: [CIC IoT-DIAD 2024 Packet-Based Dataset](http://cicresearch.ca/IOTDataset/CIC%20IoT-IDAD%20Dataset%202024/Dataset/Device%20Identification_Anomaly%20Detection%20-%20Packet%20Based%20Features/)

**Dataset Sampling**: Due to the large size (~180 CSV files), this implementation uses **5% sampling per file** to balance computational efficiency with model performance. The `merge_packet_dataset.py` script samples each CSV file individually before merging to avoid memory issues.

**Merging the packet-based dataset**:
```bash
# Recommended: 5% sampling per file
python merge_packet_dataset.py \
  --dataset-dir data/ciciot_idad_2024_packet \
  --output data/merged_packet_dataset.csv \
  --sample-fraction 0.05

# Custom sampling (e.g., 20%)
python merge_dataset_sampled.py --sample-fraction 0.2

# Specify custom paths
python merge_dataset_sampled.py --dataset-dir data/ciciot_idad_2024 --output data/merged_flow_dataset.csv
```

**Example training usage**:
```bash
# Using merged packet-based features (RECOMMENDED)
python main.py --data data/merged_packet_dataset.csv --no-optimize --no-smote

# With optimization (slower, 5 trials per model)
python main.py --data data/merged_packet_dataset.csv --no-smote

# Legacy: Using flow-based features (lower performance)
python main.py --data data/merged_flow_dataset.csv --no-optimize --no-smote
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

### Dataset Configuration (Packet-Based - 5% Sampling)
- **Dataset Type**: Packet-Based Features (RECOMMENDED)
- **Sampling**: 5% per CSV file (from 180 CSV files)
- **Total Samples**: 1,783,557 samples after merging
- **Attack Classes**: 28 granular attack types
- **Features After Preprocessing**: 96 numerical features
  - Original: 136 columns (135 features + Label)
  - Dropped: 7 identifier columns (stream, MAC/IP addresses, ports)
  - Dropped: High-cardinality categorical columns (>100 unique values)
  - Added: Statistical features (packet length std/range/CV, IAT stats, rate aggregations, squared/log transforms)
  - Removed: Constant features and highly correlated features (>0.95 correlation)
  - Final: 96 numerical features for model input

**Class Distribution (28 Attack Types):**
- **IoT Malware**:
  - Mirai-greip_flood: 304,852 (17.09%) - Largest class
  - Backdoor_Malware: 1,667 (0.09%)
- **DDoS Variants** (59.99% total):
  - DDoS-ICMP_Fragmentation: 288,682 (16.19%)
  - DDoS-UDP_Flood: 219,143 (12.29%)
  - DDoS-UDP_Fragmentation: 185,271 (10.39%)
  - DDoS-ACK_Fragmentation: 184,844 (10.36%)
  - DDoS-SynonymousIP_Flood: 138,219 (7.75%)
  - DDoS-TCP_Flood: 78,421 (4.40%)
  - DDoS-HTTP_Flood: 14,517 (0.81%)
  - DDoS-SlowLoris: 14,429 (0.81%)
- **DoS Variants** (9.83% total):
  - DoS-UDP_Flood: 77,856 (4.37%)
  - DoS-TCP_Flood: 46,645 (2.62%)
  - DoS-SYN_Flood: 34,962 (1.96%)
  - DoS-HTTP_Flood: 15,701 (0.88%)
- **Reconnaissance** (2.49% total):
  - Recon-OSScan: 14,595 (0.82%)
  - Recon-HostDiscovery: 14,418 (0.81%)
  - Recon-PortScan: 14,331 (0.80%)
  - Recon-PingSweep: 1,130 (0.06%)
- **Web Attacks** (1.19% total):
  - VulnerabilityScan: 12,946 (0.73%)
  - BrowserHijacking: 2,963 (0.17%)
  - CommandInjection: 2,781 (0.16%)
  - SqlInjection: 2,631 (0.15%)
  - XSS: 2,005 (0.11%)
  - Uploading_Attack: 646 (0.04%)
- **Network Attacks**:
  - MITM-ArpSpoofing: 29,389 (1.65%)
  - DNS_Spoofing: 14,801 (0.83%)
  - DictionaryBruteForce: 6,573 (0.37%)
- **Benign Traffic**: 59,139 (3.32%)

**Train/Validation/Test Split:**
- Training: 1,070,133 samples (60%)
- Validation: 356,712 samples (20%)
- Test: 356,712 samples (20%)

**Class Balancing**: Borderline-SMOTE / ADASYN applied to minority classes
**Optimization Trials**: 5 trials per model (DAE, CNN, MLP, LSTM)

### Models to be Evaluated

| Model | Description | Key Features |
|-------|-------------|--------------|
| **DAE** | Denoising Autoencoder | Dimensionality reduction: 62 → 32 features |
| **MLP** | Multi-Layer Perceptron | Residual connections, batch normalization, 3-4 hidden layers |
| **1D-CNN** | Convolutional Neural Network | Multi-kernel (3x3, 5x5, 7x7), parallel feature extraction |
| **BiLSTM+Attention** | Bidirectional LSTM | Self-attention mechanism, recurrent dropout |
| **Ensemble** | Stacking ensemble | XGBoost meta-learner combining CNN+MLP+LSTM predictions |

### Optimized Hyperparameters (5% Dataset, 5 Trials)

**Denoising Autoencoder (DAE) - Packet-Based Dataset (Optimized)**
- Latent dimensions: 64 (reduction from 92 input features)
- Encoder layers: [512, 256, 128]
- Noise factor: 0.3
- Dropout rate: 0.3
- Learning rate: 0.001
- Batch size: 256
- Architecture: Optimized via Optuna (5 trials, Trial 1 best)
- **Best validation loss**: 1.082e+19
- **Dimensionality reduction**: 92 features → 64 features
- Rationale: Optimal balance between compression and information preservation

**SMOTE Balancing Strategy (Packet-Based Dataset)**
- Method: Borderline-SMOTE (class-specific)
- Target: 9,145 samples per minority class (~5% of majority class Mirai: 182,911)
- **17 minority classes** balanced (60.7% of all classes)
- **11 majority classes** unchanged (Mirai, DDoS variants, DoS variants)
- Conservative approach: Minimal synthetic noise for better generalization

**Classes Balanced (17 total):**
- Uploading_Attack (class 25): 388 → 9,145 (+8,757 synthetic, 23.6x)
- Recon-PingSweep (class 22): 678 → 9,145 (+8,467 synthetic, 13.5x)
- Backdoor_Malware (class 0): 1,001 → 9,145 (+8,144 synthetic, 9.1x)
- XSS (class 27): 1,203 → 9,145 (+7,942 synthetic, 7.6x)
- BrowserHijacking (class 24): 1,579 → 9,145 (+7,566 synthetic, 5.8x)
- CommandInjection (class 3): 1,669 → 9,145 (+7,476 synthetic, 5.5x)
- SqlInjection (class 2): 1,777 → 9,145 (+7,368 synthetic, 5.1x)
- DictionaryBruteForce (class 13): 3,943 → 9,145 (+5,202 synthetic, 2.3x)
- VulnerabilityScan (class 26): 7,768 → 9,145 (+1,377 synthetic, 1.2x)
- DDoS-SlowLoris (class 7): 8,657 → 9,145 (+488 synthetic, 1.1x)
- Recon-PortScan (class 20): 8,650 → 9,145 (+495 synthetic, 1.1x)
- DDoS-HTTP_Flood (class 5): 8,710 → 9,145 (+435 synthetic, 1.0x)
- Recon-HostDiscovery (class 21): 8,757 → 9,145 (+388 synthetic, 1.0x)
- Recon-OSScan (class 23): 8,599 → 9,145 (+546 synthetic, 1.1x)
- DNS_Spoofing (class 12): 8,881 → 9,145 (+264 synthetic, 1.0x)
- DoS-HTTP_Flood (class 14): 9,421 (no upsampling - already above target)
- MITM-ArpSpoofing (class 18): 17,633 (no upsampling - already above target)

**Classes NOT Balanced (11 majority classes):**
- Mirai-greip_flood (class 19): 182,911 samples (majority class)
- DDoS-ICMP_Fragmentation (class 6): 173,210 samples
- DDoS-UDP_Flood (class 8): 131,486 samples
- DDoS-UDP_Fragmentation (class 11): 111,163 samples
- DDoS-ACK_Fragmentation (class 4): 110,906 samples
- DDoS-SynonymousIP_Flood (class 10): 82,932 samples
- DoS-UDP_Flood (class 17): 46,714 samples
- DDoS-TCP_Flood (class 9): 47,053 samples
- BenignTraffic (class 1): 35,484 samples
- DoS-TCP_Flood (class 16): 27,987 samples
- DoS-SYN_Flood (class 15): 20,977 samples

**Total Training Samples After SMOTE**: ~1,200,000 samples (increased from 1,070,133)
- 80% less synthetic data compared to 25% target (1,748,094 samples)
- Faster training and more stable validation

**1D-CNN Classifier (Optimized - Trial 1)**
- Filters: [128, 256, 512] - **Optimized via Optuna**
- Kernel sizes: [3, 5, 7]
- Dropout rate: 0.3
- Dense units: [512, 256]
- Learning rate: 0.0001 - **Lower LR for stability**
- Batch size: 128
- Early stopping patience: 25 epochs
- Loss function: Sparse categorical crossentropy (focal loss disabled)
- Class weights: Applied to handle remaining imbalance
- Architecture: Multi-kernel parallel feature extraction
- **Validation loss: 3.180** (best from 2 optimization trials)

**MLP Classifier (Improved v2)**
- Hidden layers: [256, 128] - **Simplified from [512, 256, 128]**
- Dropout rate: 0.3 - **Reduced from 0.4**
- Learning rate: 0.001
- Batch size: 128
- Early stopping patience: 25 epochs
- Activation: ReLU
- Batch normalization: Enabled
- Loss function: Sparse categorical crossentropy + class weights
- Architecture: Simplified deep feedforward network

**BiLSTM Classifier (Improved v2)**
- LSTM units: [128, 64] - **Simplified from [256, 128]**
- Dropout rate: 0.3 - **Reduced from 0.4**
- Recurrent dropout: 0.2
- Learning rate: 0.001
- Batch size: 128 - **Increased from 64**
- Early stopping patience: 25 epochs
- Bidirectional: True
- Attention mechanism: Enabled
- Timesteps: 10
- Loss function: Sparse categorical crossentropy + class weights

**Key Improvements in v2:**
- ✅ Disabled focal loss (caused training instability)
- ✅ Reduced SMOTE oversampling (50% → 25% to reduce synthetic noise)
- ✅ Increased learning rates for faster convergence
- ✅ Increased batch sizes for more stable gradients
- ✅ Increased early stopping patience for better convergence
- ✅ Simplified MLP and LSTM architectures to reduce overfitting
- ✅ Rely on class weights instead of focal loss for imbalance handling

**Expected Performance (Packet-Based, No SMOTE):**
- Overall Accuracy: 75-85% (realistic multi-class detection)
- Macro F1: 40-65% (balanced performance across all classes)
- Minority class F1: 30-60% (actual detection vs 0% with flow-based)
- CNN: 75-82% accuracy
- MLP: 72-80% accuracy
- LSTM: 73-82% accuracy
- Ensemble: 78-87% accuracy

**Note:** Results with packet-based features are significantly better than flow-based (83% accuracy but only detecting DoS class)

**Note:** Ensemble hyperparameters will be updated after training completes.

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
