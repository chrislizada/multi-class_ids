"""
IMPROVED SMOTE Balancing Module
Adaptive strategy with better minority class handling
"""

import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import optuna


class SMOTEBalancer:
    def __init__(self, config, random_state=42):
        self.config = config
        self.random_state = random_state
        self.sampler = None
        self.best_params = None
        
    def balance_data(self, X, y, method='borderline', k_neighbors=5, 
                    sampling_strategy='not majority'):
        print(f"\n" + "="*80)
        print(f"BALANCING DATA USING {method.upper()}")
        print("="*80 + "\n")
        
        print("Original class distribution:")
        original_dist = Counter(y)
        for class_label, count in sorted(original_dist.items()):
            print(f"  Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
        
        if method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
        elif method == 'borderline':
            self.sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state,
                kind='borderline-1'
            )
        elif method == 'borderline2':
            self.sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state,
                kind='borderline-2'
            )
        elif method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors,
                random_state=self.random_state
            )
        elif method == 'smote_tomek':
            self.sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                smote=SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            min_class_size = min(Counter(y).values())
            
            if k_neighbors >= min_class_size:
                k_neighbors_adj = max(1, min_class_size - 1)
                print(f"Warning: Adjusting k_neighbors from {k_neighbors} to {k_neighbors_adj} due to small minority class size")
                k_neighbors = k_neighbors_adj
                
                if method == 'smote':
                    self.sampler.set_params(k_neighbors=k_neighbors)
                elif method in ['borderline', 'borderline2']:
                    self.sampler.set_params(k_neighbors=k_neighbors)
                elif method == 'adasyn':
                    self.sampler.set_params(n_neighbors=k_neighbors)
            
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            print("\nBalanced class distribution:")
            balanced_dist = Counter(y_resampled)
            for class_label, count in sorted(balanced_dist.items()):
                original_count = original_dist.get(class_label, 0)
                change = count - original_count
                print(f"  Class {class_label}: {count} samples ({count/len(y_resampled)*100:.2f}%) "
                     f"[+{change} synthetic]")
            
            print(f"\nTotal samples: {len(y)} -> {len(y_resampled)}")
            print(f"Synthetic samples added: {len(y_resampled) - len(y)}")
            print("="*80 + "\n")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Error during resampling: {e}")
            print("Returning original data without balancing")
            return X, y
    
    def optimize_smote_params(self, X_train, y_train, X_val, y_val, 
                            classifier_builder, n_trials=10):
        print("\n" + "="*80)
        print("OPTIMIZING SMOTE HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            method = trial.suggest_categorical('method', self.config['method'])
            
            if method == 'adasyn':
                k_neighbors = trial.suggest_categorical('k_neighbors', [3, 5])
            else:
                k_neighbors = trial.suggest_categorical('k_neighbors', self.config['k_neighbors'])
            
            try:
                X_balanced, y_balanced = self.balance_data(
                    X_train, y_train,
                    method=method,
                    k_neighbors=k_neighbors,
                    sampling_strategy=self.config['sampling_strategy']
                )
                
                classifier = classifier_builder()
                classifier.fit(X_balanced, y_balanced)
                
                y_val_pred = classifier.predict(X_val)
                
                from sklearn.metrics import f1_score
                score = f1_score(y_val, y_val_pred, average='macro')
                
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nBest SMOTE parameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best F1-score: {study.best_value:.4f}")
        
        return self.best_params
    
    def apply_best_params(self, X, y):
        if self.best_params is None:
            print("Warning: No optimized parameters found. Using default borderline-SMOTE.")
            return self.balance_data(X, y, method='borderline', k_neighbors=5)
        
        return self.balance_data(
            X, y,
            method=self.best_params['method'],
            k_neighbors=self.best_params['k_neighbors'],
            sampling_strategy=self.config['sampling_strategy']
        )
    
    def balance_with_adaptive_strategy(self, X, y, minority_classes=None):
        print("\n" + "="*80)
        print("APPLYING ADAPTIVE TIERED BALANCING STRATEGY")
        print("="*80 + "\n")
        
        class_dist = Counter(y)
        n_samples = len(y)
        
        if minority_classes is None:
            mean_count = np.mean(list(class_dist.values()))
            minority_classes = [cls for cls, count in class_dist.items() 
                              if count < mean_count * 0.5]
        
        print(f"Minority classes identified: {minority_classes}")
        
        sampling_strategy = {}
        max_count = max(class_dist.values())
        
        tier1_tiny = []
        tier2_small = []
        tier3_medium = []
        
        for cls in minority_classes:
            current_count = class_dist[cls]
            if current_count < 100:
                tier1_tiny.append((cls, current_count))
            elif current_count < 1000:
                tier2_small.append((cls, current_count))
            else:
                tier3_medium.append((cls, current_count))
        
        print(f"\nTier 1 (Tiny, <100): {len(tier1_tiny)} classes")
        print(f"Tier 2 (Small, 100-1000): {len(tier2_small)} classes")
        print(f"Tier 3 (Medium, 1000+): {len(tier3_medium)} classes")
        
        for cls, count in tier1_tiny:
            target = min(int(max_count * 0.30), 3000)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 1 - Class {cls}: {count} -> {sampling_strategy[cls]}")
        
        for cls, count in tier2_small:
            target = min(int(max_count * 0.25), 2500)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 2 - Class {cls}: {count} -> {sampling_strategy[cls]}")
        
        for cls, count in tier3_medium:
            target = min(int(max_count * 0.20), 2000)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 3 - Class {cls}: {count} -> {sampling_strategy[cls]}")
        
        if not sampling_strategy:
            print("No classes need balancing. Returning original data.")
            return X, y
        
        min_class_count = min([class_dist[cls] for cls in sampling_strategy.keys()])
        k_neighbors = min(5, max(1, min_class_count - 1))
        
        print(f"\nUsing k_neighbors={k_neighbors} for SMOTE")
        
        self.sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state,
            kind='borderline-1'
        )
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            print("\nFinal class distribution:")
            final_dist = Counter(y_resampled)
            for class_label, count in sorted(final_dist.items()):
                print(f"  Class {class_label}: {count} samples ({count/len(y_resampled)*100:.2f}%)")
            
            print("="*80 + "\n")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Error during adaptive resampling: {e}")
            print("Falling back to conservative borderline-SMOTE")
            
            conservative_strategy = {cls: min(1000, target) 
                                   for cls, target in sampling_strategy.items()}
            
            self.sampler = BorderlineSMOTE(
                sampling_strategy=conservative_strategy,
                k_neighbors=3,
                random_state=self.random_state,
                kind='borderline-1'
            )
            
            try:
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                return X_resampled, y_resampled
            except:
                print("Conservative fallback also failed. Returning original data.")
                return X, y
