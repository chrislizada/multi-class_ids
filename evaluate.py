"""
Evaluation script for loading and testing trained models
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import argparse

from src.config import Config
from src.utils import MetricsCalculator


def load_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)
    
    print(f"\nLoading experiment from: {experiment_dir}\n")
    
    preprocessor = joblib.load(experiment_dir / 'preprocessor.pkl')
    
    models = {}
    
    if (experiment_dir / 'cnn').exists():
        from src.models import CNNClassifier
        cnn = CNNClassifier(input_dim=1, n_classes=Config.N_CLASSES, 
                           config=Config.CNN_CONFIG)
        cnn.load(experiment_dir / 'cnn')
        models['cnn'] = cnn
        print("✓ CNN model loaded")
    
    if (experiment_dir / 'mlp').exists():
        from src.models import MLPClassifier
        mlp = MLPClassifier(input_dim=1, n_classes=Config.N_CLASSES,
                           config=Config.MLP_CONFIG)
        mlp.load(experiment_dir / 'mlp')
        models['mlp'] = mlp
        print("✓ MLP model loaded")
    
    if (experiment_dir / 'lstm').exists():
        from src.models import LSTMClassifier
        lstm = LSTMClassifier(input_dim=1, n_classes=Config.N_CLASSES,
                             config=Config.LSTM_CONFIG)
        lstm.load(experiment_dir / 'lstm')
        models['lstm'] = lstm
        print("✓ LSTM model loaded")
    
    dae = None
    if (experiment_dir / 'dae').exists():
        from src.models import DenoisingAutoencoder
        dae = DenoisingAutoencoder(input_dim=1, config=Config.DAE_CONFIG)
        dae.load(experiment_dir / 'dae')
        print("✓ DAE model loaded")
    
    smote_balancer = None
    if (experiment_dir / 'smote_balancer.pkl').exists():
        smote_balancer = joblib.load(experiment_dir / 'smote_balancer.pkl')
        print("✓ SMOTE balancer loaded")
    
    ensemble = None
    if (experiment_dir / 'ensemble').exists() and len(models) >= 2:
        from src.models import EnsembleClassifier
        ensemble = EnsembleClassifier(
            cnn_model=models.get('cnn'),
            mlp_model=models.get('mlp'),
            lstm_model=models.get('lstm'),
            config=Config.ENSEMBLE_CONFIG,
            n_classes=Config.N_CLASSES
        )
        ensemble.load(experiment_dir / 'ensemble')
        models['ensemble'] = ensemble
        print("✓ Ensemble model loaded")
    
    print(f"\nTotal models loaded: {len(models)}\n")
    
    return preprocessor, models, dae, smote_balancer


def evaluate_on_new_data(experiment_dir, data_path, output_dir=None):
    print("\n" + "="*80)
    print("EVALUATING TRAINED MODELS ON NEW DATA")
    print("="*80 + "\n")
    
    preprocessor, models, dae, smote_balancer = load_experiment(experiment_dir)
    
    from src.preprocessing import DataPreprocessor
    print("Loading and preprocessing new data...")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    X, y = preprocessor.preprocess(df, label_column='label', fit=False)
    
    if dae is not None:
        print("Applying DAE encoding...")
        X = dae.encode(X)
    
    metrics_calc = MetricsCalculator(Config.ATTACK_CLASSES)
    
    if output_dir is None:
        output_dir = Path(experiment_dir) / 'evaluation_results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name.upper()} Model")
        print(f"{'='*80}\n")
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        metrics, per_class = metrics_calc.calculate_metrics(y, y_pred, y_proba)
        
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        metrics_calc.plot_confusion_matrix(
            y, y_pred,
            model_dir / 'confusion_matrix.png',
            title=f'{model_name.upper()} - Confusion Matrix'
        )
        
        metrics_calc.plot_normalized_confusion_matrix(
            y, y_pred,
            model_dir / 'confusion_matrix_normalized.png',
            title=f'{model_name.upper()} - Normalized Confusion Matrix'
        )
        
        metrics_calc.plot_classification_report(
            per_class,
            model_dir / 'classification_report.png',
            title=f'{model_name.upper()} - Per-Class Performance'
        )
        
        metrics_calc.plot_roc_curves(
            y, y_proba,
            model_dir / 'roc_curves.png',
            title=f'{model_name.upper()} - ROC Curves'
        )
        
        metrics_calc.save_metrics_to_json(
            metrics, per_class,
            model_dir / 'metrics.json'
        )
        
        metrics_calc.create_summary_report(
            metrics, per_class,
            model_dir / 'report.txt'
        )
        
        all_results[model_name] = metrics
    
    summary_df = pd.DataFrame({
        'Model': [name.upper() for name in all_results.keys()],
        'Accuracy': [metrics['accuracy'] for metrics in all_results.values()],
        'F1-Score (Macro)': [metrics['f1_macro'] for metrics in all_results.values()],
        'F1-Score (Weighted)': [metrics['f1_weighted'] for metrics in all_results.values()]
    })
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    print(summary_df.to_string(index=False))
    print("\n")
    
    summary_df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
    
    print(f"Evaluation results saved to: {output_dir}\n")
    
    return all_results


def compare_experiments(experiment_dirs):
    print("\n" + "="*80)
    print("COMPARING MULTIPLE EXPERIMENTS")
    print("="*80 + "\n")
    
    comparison_data = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        for model_type in ['cnn', 'mlp', 'lstm', 'ensemble']:
            metrics_file = exp_path / model_type / 'metrics.json'
            
            if metrics_file.exists():
                import json
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                comparison_data.append({
                    'Experiment': exp_name,
                    'Model': model_type.upper(),
                    'Accuracy': data['overall_metrics']['accuracy'],
                    'F1-Macro': data['overall_metrics']['f1_macro'],
                    'F1-Weighted': data['overall_metrics']['f1_weighted']
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print("\n")
        
        pivot = df.pivot(index='Experiment', columns='Model', values='Accuracy')
        print("\nAccuracy Comparison:")
        print(pivot.to_string())
        print("\n")
        
        return df
    else:
        print("No metrics found in the provided experiments.\n")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained IDS models')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to evaluation dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results = evaluate_on_new_data(
        experiment_dir=args.experiment,
        data_path=args.data,
        output_dir=args.output
    )
