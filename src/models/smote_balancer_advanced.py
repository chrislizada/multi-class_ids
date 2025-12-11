"""
ADVANCED SMOTE Balancing with LSH-SMOTE, Borderline-SMOTE, and SMOTE-ENN
Optimized for highly imbalanced multi-class IDS datasets
"""

import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import optuna


class AdvancedSMOTEBalancer:
    def __init__(self, config, random_state=42):
        self.config = config
        self.random_state = random_state
        self.sampler = None
        self.best_params = None
        
    def balance_data(self, X, y, method='borderline', k_neighbors=5, 
                    sampling_strategy='not majority', m_neighbors=10):
        print(f"\n" + "="*80)
        print(f"BALANCING DATA USING {method.upper()}")
        print("="*80 + "\n")
        
        print("Original class distribution:")
        original_dist = Counter(y)
        for class_label, count in sorted(original_dist.items()):
            print(f"  Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
        
        min_class_size = min(original_dist.values())
        k_neighbors_safe = min(k_neighbors, max(1, min_class_size - 1))
        m_neighbors_safe = min(m_neighbors, max(1, min_class_size - 1))
        
        if k_neighbors != k_neighbors_safe:
            print(f"\n⚠️  Adjusted k_neighbors: {k_neighbors} → {k_neighbors_safe} (min class: {min_class_size})")
        
        try:
            if method == 'smote':
                self.sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors_safe,
                    random_state=self.random_state
                )
            
            elif method == 'borderline':
                self.sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors_safe,
                    m_neighbors=m_neighbors_safe,
                    random_state=self.random_state,
                    kind='borderline-1'
                )
            
            elif method == 'borderline2':
                self.sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors_safe,
                    m_neighbors=m_neighbors_safe,
                    random_state=self.random_state,
                    kind='borderline-2'
                )
            
            elif method == 'adasyn':
                self.sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    n_neighbors=k_neighbors_safe,
                    random_state=self.random_state
                )
            
            elif method == 'svm_smote':
                self.sampler = SVMSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors_safe,
                    m_neighbors=m_neighbors_safe,
                    random_state=self.random_state,
                    svm_estimator=None,
                    out_step=0.5
                )
            
            elif method == 'smote_tomek':
                self.sampler = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    smote=SMOTE(k_neighbors=k_neighbors_safe, random_state=self.random_state)
                )
            
            elif method == 'smote_enn':
                self.sampler = SMOTEENN(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    smote=BorderlineSMOTE(
                        k_neighbors=k_neighbors_safe,
                        m_neighbors=m_neighbors_safe,
                        random_state=self.random_state,
                        kind='borderline-1'
                    )
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            print("\nBalanced class distribution:")
            balanced_dist = Counter(y_resampled)
            for class_label, count in sorted(balanced_dist.items()):
                original_count = original_dist.get(class_label, 0)
                change = count - original_count
                change_pct = (change / original_count * 100) if original_count > 0 else 0
                print(f"  Class {class_label}: {count} samples ({count/len(y_resampled)*100:.2f}%) "
                     f"[+{change:>6} synthetic, +{change_pct:>5.1f}%]")
            
            print(f"\nTotal samples: {len(y):>8,} → {len(y_resampled):>8,}")
            print(f"Synthetic samples added: {len(y_resampled) - len(y):>8,} ({(len(y_resampled) - len(y))/len(y)*100:.1f}%)")
            print("="*80 + "\n")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"❌ Error during {method} resampling: {e}")
            print("Returning original data without balancing")
            return X, y
    
    def balance_with_hybrid_strategy(self, X, y, minority_classes=None):
        """
        Hybrid strategy combining Borderline-SMOTE for boundary samples
        and SMOTE-ENN for noise removal
        """
        print("\n" + "="*80)
        print("APPLYING HYBRID BORDERLINE-SMOTE + ENN STRATEGY")
        print("="*80 + "\n")
        
        class_dist = Counter(y)
        
        if minority_classes is None:
            mean_count = np.mean(list(class_dist.values()))
            minority_classes = [cls for cls, count in class_dist.items() 
                              if count < mean_count * 0.5]
        
        print(f"Minority classes identified: {len(minority_classes)}")
        
        sampling_strategy = {}
        max_count = max(class_dist.values())
        
        tier1_critical = []
        tier2_severe = []
        tier3_moderate = []
        tier4_mild = []
        
        for cls in minority_classes:
            current_count = class_dist[cls]
            if current_count < 50:
                tier1_critical.append((cls, current_count))
            elif current_count < 200:
                tier2_severe.append((cls, current_count))
            elif current_count < 1000:
                tier3_moderate.append((cls, current_count))
            else:
                tier4_mild.append((cls, current_count))
        
        print(f"\n📊 TIER BREAKDOWN:")
        print(f"  Tier 1 (Critical, <50):       {len(tier1_critical):>2} classes")
        print(f"  Tier 2 (Severe, 50-200):      {len(tier2_severe):>2} classes")
        print(f"  Tier 3 (Moderate, 200-1000):  {len(tier3_moderate):>2} classes")
        print(f"  Tier 4 (Mild, 1000+):         {len(tier4_mild):>2} classes")
        
        print(f"\n🎯 TARGET CALCULATION:")
        
        for cls, count in tier1_critical:
            target = min(int(max_count * 0.35), 4000)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 1 - Class {cls:>2}: {count:>6,} → {sampling_strategy[cls]:>6,} ({sampling_strategy[cls]/count:>5.1f}x)")
        
        for cls, count in tier2_severe:
            target = min(int(max_count * 0.28), 3000)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 2 - Class {cls:>2}: {count:>6,} → {sampling_strategy[cls]:>6,} ({sampling_strategy[cls]/count:>5.1f}x)")
        
        for cls, count in tier3_moderate:
            target = min(int(max_count * 0.22), 2200)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 3 - Class {cls:>2}: {count:>6,} → {sampling_strategy[cls]:>6,} ({sampling_strategy[cls]/count:>5.1f}x)")
        
        for cls, count in tier4_mild:
            target = min(int(max_count * 0.18), 1800)
            sampling_strategy[cls] = max(target, count)
            print(f"  Tier 4 - Class {cls:>2}: {count:>6,} → {sampling_strategy[cls]:>6,} ({sampling_strategy[cls]/count:>5.1f}x)")
        
        if not sampling_strategy:
            print("\n✅ No classes need balancing. Returning original data.")
            return X, y
        
        min_class_count = min([class_dist[cls] for cls in sampling_strategy.keys()])
        k_neighbors = min(5, max(1, min_class_count - 1))
        m_neighbors = min(10, max(1, min_class_count - 1))
        
        print(f"\n⚙️  SMOTE PARAMETERS:")
        print(f"  k_neighbors: {k_neighbors}")
        print(f"  m_neighbors: {m_neighbors}")
        print(f"  Method: Borderline-SMOTE-1 + ENN")
        
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import BorderlineSMOTE
        
        self.sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            smote=BorderlineSMOTE(
                k_neighbors=k_neighbors,
                m_neighbors=m_neighbors,
                random_state=self.random_state,
                kind='borderline-1'
            )
        )
        
        try:
            print(f"\n🔄 Applying SMOTE-ENN...")
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            print("\n✅ FINAL CLASS DISTRIBUTION:")
            final_dist = Counter(y_resampled)
            for class_label, count in sorted(final_dist.items()):
                original_count = class_dist.get(class_label, 0)
                change = count - original_count
                print(f"  Class {class_label:>2}: {count:>8,} samples ({count/len(y_resampled)*100:>5.2f}%) "
                      f"[{change:>+8,}]")
            
            print(f"\n📈 SUMMARY:")
            print(f"  Original samples:  {len(y):>10,}")
            print(f"  Final samples:     {len(y_resampled):>10,}")
            print(f"  Net change:        {len(y_resampled) - len(y):>+10,} ({(len(y_resampled) - len(y))/len(y)*100:>+6.1f}%)")
            print("="*80 + "\n")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"\n❌ Error during hybrid resampling: {e}")
            print("Trying conservative fallback...")
            
            conservative_strategy = {}
            for cls, target in sampling_strategy.items():
                conservative_target = min(target, class_dist[cls] * 3, 1500)
                if conservative_target > class_dist[cls]:
                    conservative_strategy[cls] = conservative_target
            
            if not conservative_strategy:
                print("No conservative strategy possible. Returning original data.")
                return X, y
            
            try:
                self.sampler = BorderlineSMOTE(
                    sampling_strategy=conservative_strategy,
                    k_neighbors=3,
                    m_neighbors=5,
                    random_state=self.random_state,
                    kind='borderline-1'
                )
                
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                print("✅ Conservative fallback succeeded")
                return X_resampled, y_resampled
                
            except Exception as e2:
                print(f"❌ Conservative fallback also failed: {e2}")
                print("Returning original data without balancing")
                return X, y
    
    def optimize_smote_params(self, X_train, y_train, X_val, y_val, 
                            classifier_builder, n_trials=10):
        print("\n" + "="*80)
        print("OPTIMIZING SMOTE HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            method = trial.suggest_categorical(
                'method', 
                ['borderline', 'borderline2', 'smote_enn', 'adasyn']
            )
            
            k_neighbors = trial.suggest_int('k_neighbors', 3, 7)
            m_neighbors = trial.suggest_int('m_neighbors', 5, 15)
            
            try:
                X_balanced, y_balanced = self.balance_data(
                    X_train, y_train,
                    method=method,
                    k_neighbors=k_neighbors,
                    m_neighbors=m_neighbors,
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
        
        print(f"\n✅ BEST SMOTE PARAMETERS:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best F1-score: {study.best_value:.4f}")
        
        return self.best_params
    
    def apply_best_params(self, X, y):
        if self.best_params is None:
            print("⚠️  No optimized parameters. Using hybrid strategy.")
            return self.balance_with_hybrid_strategy(X, y)
        
        return self.balance_data(
            X, y,
            method=self.best_params['method'],
            k_neighbors=self.best_params.get('k_neighbors', 5),
            m_neighbors=self.best_params.get('m_neighbors', 10),
            sampling_strategy=self.config['sampling_strategy']
        )
