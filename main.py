"""
Main Training Pipeline for Multi-Class IDS
Integrates all components: preprocessing, DAE, SMOTE, classifiers, and ensemble
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from datetime import datetime
import joblib

from src.config import Config
from src.preprocessing import DataPreprocessor
from src.models import (
    DenoisingAutoencoder,
    SMOTEBalancer,
    CNNClassifier,
    MLPClassifier,
    LSTMClassifier,
    EnsembleClassifier
)
from src.utils import MetricsCalculator, calculate_class_weights


def main(data_path, optimize_hyperparams=True, use_dae=True, use_smote=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Config.RESULTS_DIR / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MULTI-CLASS INTRUSION DETECTION SYSTEM")
    print("Enhanced with DAE, SMOTE, and Deep Learning Classifiers")
    print("="*80 + "\n")
    
    print(f"Experiment directory: {experiment_dir}\n")
    
    preprocessor = DataPreprocessor(Config)
    
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    df = preprocessor.load_data(data_path)
    
    X, y = preprocessor.preprocess(df, label_column='label', fit=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        test_size=Config.TEST_SIZE,
        val_size=Config.VALIDATION_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    joblib.dump(preprocessor, experiment_dir / 'preprocessor.pkl')
    
    if use_dae:
        print("\n" + "="*80)
        print("STEP 2: DIMENSIONALITY REDUCTION WITH DAE")
        print("="*80)
        
        dae = DenoisingAutoencoder(input_dim=X_train.shape[1], config=Config.DAE_CONFIG)
        
        if optimize_hyperparams:
            dae.optimize_hyperparameters(X_train, X_val, n_trials=20)
        else:
            dae.build_autoencoder(
                encoder_layers=[512, 256],
                latent_dim=64,
                dropout_rate=0.3,
                learning_rate=0.001
            )
            dae.train(X_train, X_val, noise_factor=0.2, batch_size=128, 
                     epochs=50, patience=10)
        
        X_train_encoded = dae.encode(X_train)
        X_val_encoded = dae.encode(X_val)
        X_test_encoded = dae.encode(X_test)
        
        dae.save(experiment_dir / 'dae')
        
        print(f"\nDimensionality reduced: {X_train.shape[1]} -> {X_train_encoded.shape[1]}")
        
        X_train = X_train_encoded
        X_val = X_val_encoded
        X_test = X_test_encoded
    
    class_weights_original = calculate_class_weights(y_train)
    
    if use_smote:
        print("\n" + "="*80)
        print("STEP 3: CLASS BALANCING WITH SMOTE")
        print("="*80)
        
        smote_balancer = SMOTEBalancer(Config.SMOTE_CONFIG, random_state=Config.RANDOM_STATE)
        
        from collections import Counter
        class_dist = Counter(y_train)
        mean_count = np.mean(list(class_dist.values()))
        minority_classes = [cls for cls, count in class_dist.items() if count < mean_count * 0.5]
        
        print(f"Auto-detected minority classes: {minority_classes}")
        
        X_train, y_train = smote_balancer.balance_with_class_specific_strategy(
            X_train, y_train, minority_classes=minority_classes if len(minority_classes) > 0 else None
        )
        
        joblib.dump(smote_balancer, experiment_dir / 'smote_balancer.pkl')
    
    class_weights = class_weights_original
    print(f"\nClass weights: {class_weights}")
    
    metrics_calc = MetricsCalculator(Config.ATTACK_CLASSES)
    
    print("\n" + "="*80)
    print("STEP 4: TRAINING CLASSIFIERS")
    print("="*80 + "\n")
    
    results = {}
    
    print("\n" + "-"*80)
    print("4.1 TRAINING CNN CLASSIFIER")
    print("-"*80)
    
    cnn = CNNClassifier(
        input_dim=X_train.shape[1],
        n_classes=Config.N_CLASSES,
        config=Config.CNN_CONFIG,
        use_focal_loss=True
    )
    
    if optimize_hyperparams:
        cnn.optimize_hyperparameters(X_train, y_train, X_val, y_val, 
                                    class_weight=class_weights, n_trials=20)
    else:
        cnn.build_model(filters=[64, 128, 256], kernel_sizes=[3, 5, 7],
                       dropout_rate=0.4, dense_units=[256, 128], learning_rate=0.001)
        cnn.train(X_train, y_train, X_val, y_val, batch_size=64, 
                 epochs=100, patience=15, class_weight=class_weights)
    
    y_test_pred_cnn = cnn.predict(X_test)
    y_test_proba_cnn = cnn.predict_proba(X_test)
    
    metrics_cnn, per_class_cnn = metrics_calc.calculate_metrics(
        y_test, y_test_pred_cnn, y_test_proba_cnn
    )
    
    cnn_dir = experiment_dir / 'cnn'
    cnn_dir.mkdir(exist_ok=True)
    cnn.save(cnn_dir)
    
    metrics_calc.plot_confusion_matrix(
        y_test, y_test_pred_cnn, 
        cnn_dir / 'confusion_matrix.png',
        title='CNN - Confusion Matrix'
    )
    metrics_calc.plot_normalized_confusion_matrix(
        y_test, y_test_pred_cnn,
        cnn_dir / 'confusion_matrix_normalized.png',
        title='CNN - Normalized Confusion Matrix'
    )
    metrics_calc.plot_classification_report(
        per_class_cnn, cnn_dir / 'classification_report.png',
        title='CNN - Per-Class Performance'
    )
    metrics_calc.plot_training_history(
        cnn.history.history, cnn_dir / 'training_history.png',
        title='CNN - Training History'
    )
    metrics_calc.plot_roc_curves(
        y_test, y_test_proba_cnn, cnn_dir / 'roc_curves.png',
        title='CNN - ROC Curves'
    )
    metrics_calc.save_metrics_to_json(
        metrics_cnn, per_class_cnn, cnn_dir / 'metrics.json'
    )
    metrics_calc.create_summary_report(
        metrics_cnn, per_class_cnn, cnn_dir / 'report.txt'
    )
    
    results['cnn'] = metrics_cnn
    
    print("\n" + "-"*80)
    print("4.2 TRAINING MLP CLASSIFIER")
    print("-"*80)
    
    mlp = MLPClassifier(
        input_dim=X_train.shape[1],
        n_classes=Config.N_CLASSES,
        config=Config.MLP_CONFIG,
        use_focal_loss=True
    )
    
    if optimize_hyperparams:
        mlp.optimize_hyperparameters(X_train, y_train, X_val, y_val, 
                                    class_weight=class_weights, n_trials=20)
    else:
        mlp.build_model(hidden_layers=[512, 256, 128], dropout_rate=0.4,
                       learning_rate=0.001, activation='relu', batch_norm=True)
        mlp.train(X_train, y_train, X_val, y_val, batch_size=128, 
                 epochs=100, patience=15, class_weight=class_weights)
    
    y_test_pred_mlp = mlp.predict(X_test)
    y_test_proba_mlp = mlp.predict_proba(X_test)
    
    metrics_mlp, per_class_mlp = metrics_calc.calculate_metrics(
        y_test, y_test_pred_mlp, y_test_proba_mlp
    )
    
    mlp_dir = experiment_dir / 'mlp'
    mlp_dir.mkdir(exist_ok=True)
    mlp.save(mlp_dir)
    
    metrics_calc.plot_confusion_matrix(
        y_test, y_test_pred_mlp, 
        mlp_dir / 'confusion_matrix.png',
        title='MLP - Confusion Matrix'
    )
    metrics_calc.plot_normalized_confusion_matrix(
        y_test, y_test_pred_mlp,
        mlp_dir / 'confusion_matrix_normalized.png',
        title='MLP - Normalized Confusion Matrix'
    )
    metrics_calc.plot_classification_report(
        per_class_mlp, mlp_dir / 'classification_report.png',
        title='MLP - Per-Class Performance'
    )
    metrics_calc.plot_training_history(
        mlp.history.history, mlp_dir / 'training_history.png',
        title='MLP - Training History'
    )
    metrics_calc.plot_roc_curves(
        y_test, y_test_proba_mlp, mlp_dir / 'roc_curves.png',
        title='MLP - ROC Curves'
    )
    metrics_calc.save_metrics_to_json(
        metrics_mlp, per_class_mlp, mlp_dir / 'metrics.json'
    )
    metrics_calc.create_summary_report(
        metrics_mlp, per_class_mlp, mlp_dir / 'report.txt'
    )
    
    results['mlp'] = metrics_mlp
    
    print("\n" + "-"*80)
    print("4.3 TRAINING LSTM CLASSIFIER")
    print("-"*80)
    
    lstm = LSTMClassifier(
        input_dim=X_train.shape[1],
        n_classes=Config.N_CLASSES,
        config=Config.LSTM_CONFIG,
        use_focal_loss=True
    )
    
    if optimize_hyperparams:
        lstm.optimize_hyperparameters(X_train, y_train, X_val, y_val, 
                                     class_weight=class_weights, n_trials=15)
    else:
        lstm.build_model(units=[256, 128], dropout_rate=0.4, recurrent_dropout=0.2,
                        learning_rate=0.001, bidirectional=True, use_attention=True)
        lstm.train(X_train, y_train, X_val, y_val, batch_size=64, 
                  epochs=100, patience=15, class_weight=class_weights)
    
    y_test_pred_lstm = lstm.predict(X_test)
    y_test_proba_lstm = lstm.predict_proba(X_test)
    
    metrics_lstm, per_class_lstm = metrics_calc.calculate_metrics(
        y_test, y_test_pred_lstm, y_test_proba_lstm
    )
    
    lstm_dir = experiment_dir / 'lstm'
    lstm_dir.mkdir(exist_ok=True)
    lstm.save(lstm_dir)
    
    metrics_calc.plot_confusion_matrix(
        y_test, y_test_pred_lstm, 
        lstm_dir / 'confusion_matrix.png',
        title='LSTM - Confusion Matrix'
    )
    metrics_calc.plot_normalized_confusion_matrix(
        y_test, y_test_pred_lstm,
        lstm_dir / 'confusion_matrix_normalized.png',
        title='LSTM - Normalized Confusion Matrix'
    )
    metrics_calc.plot_classification_report(
        per_class_lstm, lstm_dir / 'classification_report.png',
        title='LSTM - Per-Class Performance'
    )
    metrics_calc.plot_training_history(
        lstm.history.history, lstm_dir / 'training_history.png',
        title='LSTM - Training History'
    )
    metrics_calc.plot_roc_curves(
        y_test, y_test_proba_lstm, lstm_dir / 'roc_curves.png',
        title='LSTM - ROC Curves'
    )
    metrics_calc.save_metrics_to_json(
        metrics_lstm, per_class_lstm, lstm_dir / 'metrics.json'
    )
    metrics_calc.create_summary_report(
        metrics_lstm, per_class_lstm, lstm_dir / 'report.txt'
    )
    
    results['lstm'] = metrics_lstm
    
    print("\n" + "="*80)
    print("STEP 5: ENSEMBLE MODEL")
    print("="*80 + "\n")
    
    ensemble = EnsembleClassifier(
        cnn_model=cnn,
        mlp_model=mlp,
        lstm_model=lstm,
        config=Config.ENSEMBLE_CONFIG,
        n_classes=Config.N_CLASSES
    )
    
    ensemble.evaluate_individual_models(X_test, y_test)
    
    if optimize_hyperparams:
        ensemble.optimize_weights(X_val, y_val, n_trials=30)
    
    if Config.ENSEMBLE_CONFIG['use_stacking']:
        ensemble.train_stacking(X_train, y_train, X_val, y_val)
    
    y_test_pred_ensemble = ensemble.predict(X_test)
    y_test_proba_ensemble = ensemble.predict_proba(X_test)
    
    metrics_ensemble, per_class_ensemble = metrics_calc.calculate_metrics(
        y_test, y_test_pred_ensemble, y_test_proba_ensemble
    )
    
    ensemble_dir = experiment_dir / 'ensemble'
    ensemble_dir.mkdir(exist_ok=True)
    ensemble.save(ensemble_dir)
    
    metrics_calc.plot_confusion_matrix(
        y_test, y_test_pred_ensemble, 
        ensemble_dir / 'confusion_matrix.png',
        title='Ensemble - Confusion Matrix'
    )
    metrics_calc.plot_normalized_confusion_matrix(
        y_test, y_test_pred_ensemble,
        ensemble_dir / 'confusion_matrix_normalized.png',
        title='Ensemble - Normalized Confusion Matrix'
    )
    metrics_calc.plot_classification_report(
        per_class_ensemble, ensemble_dir / 'classification_report.png',
        title='Ensemble - Per-Class Performance'
    )
    metrics_calc.plot_roc_curves(
        y_test, y_test_proba_ensemble, ensemble_dir / 'roc_curves.png',
        title='Ensemble - ROC Curves'
    )
    metrics_calc.save_metrics_to_json(
        metrics_ensemble, per_class_ensemble, ensemble_dir / 'metrics.json'
    )
    metrics_calc.create_summary_report(
        metrics_ensemble, per_class_ensemble, ensemble_dir / 'report.txt'
    )
    
    results['ensemble'] = metrics_ensemble
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    import pandas as pd
    
    summary_df = pd.DataFrame({
        'Model': ['CNN', 'MLP', 'LSTM', 'Ensemble'],
        'Accuracy': [
            results['cnn']['accuracy'],
            results['mlp']['accuracy'],
            results['lstm']['accuracy'],
            results['ensemble']['accuracy']
        ],
        'F1-Score (Macro)': [
            results['cnn']['f1_macro'],
            results['mlp']['f1_macro'],
            results['lstm']['f1_macro'],
            results['ensemble']['f1_macro']
        ],
        'F1-Score (Weighted)': [
            results['cnn']['f1_weighted'],
            results['mlp']['f1_weighted'],
            results['lstm']['f1_weighted'],
            results['ensemble']['f1_weighted']
        ]
    })
    
    print(summary_df.to_string(index=False))
    print("\n")
    
    summary_df.to_csv(experiment_dir / 'summary_results.csv', index=False)
    
    best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = summary_df['Accuracy'].max()
    
    print(f"Best performing model: {best_model} (Accuracy: {best_accuracy:.4f})")
    print(f"\nAll results saved to: {experiment_dir}")
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return results, experiment_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Class IDS Training Pipeline')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to the dataset (CSV or Parquet)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--no-dae', action='store_true',
                       help='Skip DAE dimensionality reduction')
    parser.add_argument('--no-smote', action='store_true',
                       help='Skip SMOTE balancing')
    
    args = parser.parse_args()
    
    results, exp_dir = main(
        data_path=args.data,
        optimize_hyperparams=not args.no_optimize,
        use_dae=not args.no_dae,
        use_smote=not args.no_smote
    )
