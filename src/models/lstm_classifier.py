"""
BiLSTM Classifier with Attention Mechanism and Hyperparameter Optimization
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from pathlib import Path
import joblib


class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=-1)
        weighted_input = x * tf.expand_dims(a, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


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


class LSTMClassifier:
    def __init__(self, input_dim, n_classes, config, use_focal_loss=True):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.config = config
        self.use_focal_loss = use_focal_loss
        self.model = None
        self.history = None
        self.best_params = None
        self.timesteps = 10
        
    def reshape_for_lstm(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        if n_features % self.timesteps != 0:
            pad_size = self.timesteps - (n_features % self.timesteps)
            X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant')
            n_features = X.shape[1]
        
        features_per_timestep = n_features // self.timesteps
        X_reshaped = X.reshape(n_samples, self.timesteps, features_per_timestep)
        
        return X_reshaped
    
    def build_model(self, units=[256, 128], dropout_rate=0.4, recurrent_dropout=0.2,
                   learning_rate=0.001, bidirectional=True, use_attention=True):
        # Calculate features per timestep based on input_dim
        n_features = self.input_dim
        if n_features % self.timesteps != 0:
            pad_size = self.timesteps - (n_features % self.timesteps)
            n_features = n_features + pad_size
        features_per_timestep = n_features // self.timesteps
        input_shape = (self.timesteps, features_per_timestep)
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        for i, unit in enumerate(units):
            return_sequences = (i < len(units) - 1) or use_attention
            
            lstm_layer = layers.LSTM(
                unit,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout
            )
            
            if bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)
            
            x = layers.BatchNormalization()(x)
        
        if use_attention and len(x.shape) == 3:
            x = AttentionLayer()(x)
        elif len(x.shape) == 3:
            x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='bilstm_classifier')
        
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
             batch_size=64, epochs=100, patience=15, class_weight=None, verbose=1):
        print(f"\nTraining BiLSTM Classifier...")
        
        X_train_reshaped = self.reshape_for_lstm(X_train)
        
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
        
        if X_val is not None:
            X_val_reshaped = self.reshape_for_lstm(X_val)
            validation_data = (X_val_reshaped, y_val)
        else:
            validation_data = None
        
        self.history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        X_reshaped = self.reshape_for_lstm(X)
        predictions = self.model.predict(X_reshaped, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        X_reshaped = self.reshape_for_lstm(X)
        return self.model.predict(X_reshaped, verbose=0)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, 
                                class_weight=None, n_trials=20):
        print("\n" + "="*80)
        print("OPTIMIZING LSTM HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            import tensorflow as tf
            units = trial.suggest_categorical(
                'units',
                [str(u) for u in self.config['units']]
            )
            units = eval(units)
            
            dropout_rate = trial.suggest_categorical('dropout_rate', self.config['dropout_rate'])
            recurrent_dropout = trial.suggest_categorical('recurrent_dropout', self.config['recurrent_dropout'])
            learning_rate = trial.suggest_categorical('learning_rate', self.config['learning_rate'])
            batch_size = trial.suggest_categorical('batch_size', self.config['batch_size'])
            use_attention = trial.suggest_categorical('use_attention', self.config['use_attention'])
            bidirectional = trial.suggest_categorical('bidirectional', self.config['bidirectional'])
            
            self.build_model(
                units=units,
                dropout_rate=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                learning_rate=learning_rate,
                bidirectional=bidirectional,
                use_attention=use_attention
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
        self.best_params['units'] = eval(self.best_params['units'])
        
        print(f"\nBest LSTM hyperparameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        
        self.build_model(
            units=self.best_params['units'],
            dropout_rate=self.best_params['dropout_rate'],
            recurrent_dropout=self.best_params['recurrent_dropout'],
            learning_rate=self.best_params['learning_rate'],
            bidirectional=self.best_params['bidirectional'],
            use_attention=self.best_params['use_attention']
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
        
        self.model.save(save_dir / 'lstm_model.h5')
        
        if self.best_params is not None:
            joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        if self.history is not None:
            joblib.dump(self.history.history, save_dir / 'history.pkl')
        
        joblib.dump(self.timesteps, save_dir / 'timesteps.pkl')
        
        print(f"LSTM model saved to {save_dir}")
    
    def load(self, save_dir):
        save_dir = Path(save_dir)
        
        self.model = keras.models.load_model(
            save_dir / 'lstm_model.h5',
            custom_objects={'FocalLoss': FocalLoss, 'AttentionLayer': AttentionLayer}
        )
        
        if (save_dir / 'best_params.pkl').exists():
            self.best_params = joblib.load(save_dir / 'best_params.pkl')
        
        if (save_dir / 'history.pkl').exists():
            history_dict = joblib.load(save_dir / 'history.pkl')
            
            class HistoryWrapper:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            self.history = HistoryWrapper(history_dict)
        
        if (save_dir / 'timesteps.pkl').exists():
            self.timesteps = joblib.load(save_dir / 'timesteps.pkl')
        
        print(f"LSTM model loaded from {save_dir}")
