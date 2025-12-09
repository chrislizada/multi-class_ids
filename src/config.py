"""
Configuration file for Multi-Class IDS
Contains all hyperparameters and settings
"""

import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    N_FOLDS = 5
    
    ATTACK_CLASSES = [
        'Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 
        'Recon', 'Spoofing', 'Web-Based'
    ]
    N_CLASSES = len(ATTACK_CLASSES)
    
    DAE_CONFIG = {
        'latent_dims': [32, 64, 128, 256],
        'encoder_layers': [[256, 128], [512, 256, 128], [1024, 512, 256]],
        'noise_factor': [0.1, 0.2, 0.3],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [128, 256],
        'epochs': 50,
        'patience': 10,
        'activation': 'relu',
        'optimizer': 'adam'
    }
    
    SMOTE_CONFIG = {
        'method': ['borderline', 'adasyn', 'smote'],
        'k_neighbors': [3, 5, 7],
        'sampling_strategy': 'not majority'
    }
    
    CNN_CONFIG = {
        'filters': [[64, 128, 256], [128, 256, 512]],
        'kernel_sizes': [[3, 5, 7], [5, 7, 9]],
        'dropout_rate': [0.3, 0.4, 0.5],
        'dense_units': [[256, 128], [512, 256]],
        'learning_rate': [0.0005, 0.0001],  # Lower LR to prevent NaN
        'batch_size': [64, 128],
        'epochs': 100,
        'patience': 15
    }
    
    MLP_CONFIG = {
        'hidden_layers': [[512, 256, 128], [1024, 512, 256], [1024, 512, 256, 128]],
        'dropout_rate': [0.3, 0.4, 0.5],
        'learning_rate': [0.0005, 0.0001, 0.00001],  # Lower LR to prevent NaN
        'batch_size': [64, 128, 256],
        'activation': ['relu', 'selu'],
        'batch_norm': [True, False],
        'epochs': 100,
        'patience': 15
    }
    
    LSTM_CONFIG = {
        'units': [[128, 64], [256, 128], [512, 256]],
        'dropout_rate': [0.3, 0.4, 0.5],
        'recurrent_dropout': [0.2, 0.3],
        'learning_rate': [0.0005, 0.0001],  # Lower LR to prevent NaN
        'batch_size': [64, 128],
        'use_attention': [True],
        'bidirectional': [True],
        'epochs': 100,
        'patience': 15
    }
    
    ENSEMBLE_CONFIG = {
        'weights': {
            'cnn': 0.4,
            'mlp': 0.4,
            'lstm': 0.2
        },
        'voting': 'soft',
        'use_stacking': True,
        'meta_learner': 'xgboost'
    }
    
    TRAINING_CONFIG = {
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_class_weights': True,
        'use_early_stopping': True,
        'use_reduce_lr': True,
        'lr_reduce_factor': 0.5,
        'lr_reduce_patience': 5,
        'verbose': 1
    }
    
    OPTIMIZATION_CONFIG = {
        'method': 'optuna',
        'n_trials': 5,
        'n_jobs': -1,
        'timeout': 3600
    }
