"""
Ensemble Model combining CNN, MLP, and LSTM classifiers
Supports weighted voting and stacking approaches
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path


class EnsembleClassifier:
    def __init__(self, cnn_model, mlp_model, lstm_model, config, n_classes):
        self.cnn_model = cnn_model
        self.mlp_model = mlp_model
        self.lstm_model = lstm_model
        self.config = config
        self.n_classes = n_classes
        self.meta_learner = None
        self.weights = config['weights']
        self.use_stacking = config['use_stacking']
        
    def predict_proba_all_models(self, X):
        cnn_proba = self.cnn_model.predict_proba(X)
        mlp_proba = self.mlp_model.predict_proba(X)
        lstm_proba = self.lstm_model.predict_proba(X)
        
        return cnn_proba, mlp_proba, lstm_proba
    
    def weighted_voting(self, X):
        cnn_proba, mlp_proba, lstm_proba = self.predict_proba_all_models(X)
        
        weighted_proba = (
            self.weights['cnn'] * cnn_proba +
            self.weights['mlp'] * mlp_proba +
            self.weights['lstm'] * lstm_proba
        )
        
        predictions = np.argmax(weighted_proba, axis=1)
        
        return predictions, weighted_proba
    
    def train_stacking(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*80)
        print("TRAINING STACKING META-LEARNER")
        print("="*80 + "\n")
        
        print("Generating base model predictions on training set...")
        cnn_train_proba = self.cnn_model.predict_proba(X_train)
        mlp_train_proba = self.mlp_model.predict_proba(X_train)
        lstm_train_proba = self.lstm_model.predict_proba(X_train)
        
        X_train_meta = np.hstack([cnn_train_proba, mlp_train_proba, lstm_train_proba])
        
        print("Generating base model predictions on validation set...")
        cnn_val_proba = self.cnn_model.predict_proba(X_val)
        mlp_val_proba = self.mlp_model.predict_proba(X_val)
        lstm_val_proba = self.lstm_model.predict_proba(X_val)
        
        X_val_meta = np.hstack([cnn_val_proba, mlp_val_proba, lstm_val_proba])
        
        if self.config['meta_learner'] == 'xgboost':
            print("Training XGBoost meta-learner...")
            self.meta_learner = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        else:
            print("Training Logistic Regression meta-learner...")
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        
        self.meta_learner.fit(X_train_meta, y_train)
        
        y_val_pred = self.meta_learner.predict(X_val_meta)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        print(f"\nMeta-learner validation performance:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  F1-Score (macro): {val_f1:.4f}")
        print("="*80 + "\n")
        
        return self.meta_learner
    
    def predict_stacking(self, X):
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_stacking first.")
        
        cnn_proba, mlp_proba, lstm_proba = self.predict_proba_all_models(X)
        
        X_meta = np.hstack([cnn_proba, mlp_proba, lstm_proba])
        
        predictions = self.meta_learner.predict(X_meta)
        probabilities = self.meta_learner.predict_proba(X_meta)
        
        return predictions, probabilities
    
    def predict(self, X):
        if self.use_stacking and self.meta_learner is not None:
            predictions, _ = self.predict_stacking(X)
        else:
            predictions, _ = self.weighted_voting(X)
        
        return predictions
    
    def predict_proba(self, X):
        if self.use_stacking and self.meta_learner is not None:
            _, probabilities = self.predict_stacking(X)
        else:
            _, probabilities = self.weighted_voting(X)
        
        return probabilities
    
    def optimize_weights(self, X_val, y_val, n_trials=50):
        print("\n" + "="*80)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*80 + "\n")
        
        import optuna
        
        cnn_proba, mlp_proba, lstm_proba = self.predict_proba_all_models(X_val)
        
        def objective(trial):
            w_cnn = trial.suggest_float('w_cnn', 0.0, 1.0)
            w_mlp = trial.suggest_float('w_mlp', 0.0, 1.0)
            w_lstm = trial.suggest_float('w_lstm', 0.0, 1.0)
            
            total = w_cnn + w_mlp + w_lstm
            if total == 0:
                return 0.0
            
            w_cnn /= total
            w_mlp /= total
            w_lstm /= total
            
            weighted_proba = w_cnn * cnn_proba + w_mlp * mlp_proba + w_lstm * lstm_proba
            predictions = np.argmax(weighted_proba, axis=1)
            
            f1 = f1_score(y_val, predictions, average='macro')
            
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        total = best_params['w_cnn'] + best_params['w_mlp'] + best_params['w_lstm']
        
        self.weights = {
            'cnn': best_params['w_cnn'] / total,
            'mlp': best_params['w_mlp'] / total,
            'lstm': best_params['w_lstm'] / total
        }
        
        print(f"\nOptimized ensemble weights:")
        print(f"  CNN: {self.weights['cnn']:.3f}")
        print(f"  MLP: {self.weights['mlp']:.3f}")
        print(f"  LSTM: {self.weights['lstm']:.3f}")
        print(f"  Best F1-score: {study.best_value:.4f}")
        print("="*80 + "\n")
        
        return self.weights
    
    def get_individual_predictions(self, X):
        cnn_pred = self.cnn_model.predict(X)
        mlp_pred = self.mlp_model.predict(X)
        lstm_pred = self.lstm_model.predict(X)
        
        return {
            'cnn': cnn_pred,
            'mlp': mlp_pred,
            'lstm': lstm_pred
        }
    
    def evaluate_individual_models(self, X, y):
        individual_preds = self.get_individual_predictions(X)
        
        results = {}
        for model_name, predictions in individual_preds.items():
            accuracy = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average='macro')
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
        
        print("\n" + "="*80)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*80)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print("="*80 + "\n")
        
        return results
    
    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.weights, save_dir / 'weights.pkl')
        
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, save_dir / 'meta_learner.pkl')
        
        config_to_save = {
            'use_stacking': self.use_stacking,
            'n_classes': self.n_classes
        }
        joblib.dump(config_to_save, save_dir / 'ensemble_config.pkl')
        
        print(f"Ensemble model saved to {save_dir}")
    
    def load(self, save_dir):
        save_dir = Path(save_dir)
        
        if (save_dir / 'weights.pkl').exists():
            self.weights = joblib.load(save_dir / 'weights.pkl')
        
        if (save_dir / 'meta_learner.pkl').exists():
            self.meta_learner = joblib.load(save_dir / 'meta_learner.pkl')
        
        if (save_dir / 'ensemble_config.pkl').exists():
            config = joblib.load(save_dir / 'ensemble_config.pkl')
            self.use_stacking = config['use_stacking']
            self.n_classes = config['n_classes']
        
        print(f"Ensemble model loaded from {save_dir}")
