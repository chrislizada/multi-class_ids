"""
MLP Classifier with Hyperparameter Optimization
Implements deep feedforward neural network with residual connections
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from pathlib import Path
import joblib


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot if needed
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        elif y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)


class MLPClassifier:
    def __init__(self, input_dim, n_classes, config, use_focal_loss=True):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.config = config
        self.use_focal_loss = use_focal_loss
        self.model = None
        self.history = None
        self.best_params = None
        
    def build_model(self, hidden_layers=[512, 256, 128], dropout_rate=0.4,
                   learning_rate=0.001, activation='relu', batch_norm=True):
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        for i, units in enumerate(hidden_layers):
            residual = x if x.shape[-1] == units else layers.Dense(units)(x)
            
            x = layers.Dense(units, activation=None)(x)
            
            if batch_norm:
                x = layers.BatchNormalization()(x)
            
            if activation == 'relu':
                x = layers.ReLU()(x)
            elif activation == 'selu':
                x = layers.Activation('selu')(x)
            elif activation == 'prelu':
                x = layers.PReLU()(x)
            else:
                x = layers.Activation(activation)(x)
            
            x = layers.Dropout(dropout_rate)(x)
            
            if i > 0 and x.shape[-1] == residual.shape[-1]:
                x = layers.Add()([x, residual])
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='mlp_classifier')
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.use_focal_loss:
            loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
             batch_size=128, epochs=100, patience=15, class_weight=None, verbose=1):
        print(f"\nTraining MLP Classifier...")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=int(patience/3),
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, 
                                class_weight=None, n_trials=30):
        print("\n" + "="*80)
        print("OPTIMIZING MLP HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            import tensorflow as tf
            hidden_layers = trial.suggest_categorical(
                'hidden_layers',
                [str(h) for h in self.config['hidden_layers']]
            )
            hidden_layers = eval(hidden_layers)
            
            dropout_rate = trial.suggest_categorical('dropout_rate', self.config['dropout_rate'])
            learning_rate = trial.suggest_categorical('learning_rate', self.config['learning_rate'])
            batch_size = trial.suggest_categorical('batch_size', self.config['batch_size'])
            activation = trial.suggest_categorical('activation', self.config['activation'])
            batch_norm = trial.suggest_categorical('batch_norm', self.config['batch_norm'])
            
            self.build_model(
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                activation=activation,
                batch_norm=batch_norm
            )
            
            history = self.train(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size,
                epochs=self.config['epochs'],
                patience=self.config['patience'],
                class_weight=class_weight,
                verbose=0
            )
            
            val_loss = min(history.history['val_loss'])
            
            tf.keras.backend.clear_session()
            
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_params['hidden_layers'] = eval(self.best_params['hidden_layers'])
        
        print(f"\nBest MLP hyperparameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        
        self.build_model(
            hidden_layers=self.best_params['hidden_layers'],
            dropout_rate=self.best_params['dropout_rate'],
            learning_rate=self.best_params['learning_rate'],
            activation=self.best_params['activation'],
            batch_norm=self.best_params['batch_norm']
        )
        
        self.train(
            X_train, y_train, X_val, y_val,
            batch_size=self.best_params['batch_size'],
            epochs=self.config['epochs'],
            patience=self.config['patience'],
            class_weight=class_weight,
            verbose=1
        )
        
        return self.best_params
    
    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_dir / 'mlp_model.h5')
        
        if self.best_params is not None:
            joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        if self.history is not None:
            joblib.dump(self.history.history, save_dir / 'history.pkl')
        
        print(f"MLP model saved to {save_dir}")
    
    def load(self, save_dir):
        save_dir = Path(save_dir)
        
        self.model = keras.models.load_model(
            save_dir / 'mlp_model.h5',
            custom_objects={'FocalLoss': FocalLoss}
        )
        
        if (save_dir / 'best_params.pkl').exists():
            self.best_params = joblib.load(save_dir / 'best_params.pkl')
        
        if (save_dir / 'history.pkl').exists():
            history_dict = joblib.load(save_dir / 'history.pkl')
            
            class HistoryWrapper:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            self.history = HistoryWrapper(history_dict)
        
        print(f"MLP model loaded from {save_dir}")
