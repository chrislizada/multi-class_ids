"""
FIXED 1D-CNN Classifier with Improved Architecture
Addresses training collapse, gradient issues, and minority class detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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


class CNNClassifier:
    def __init__(self, input_dim, n_classes, config, use_focal_loss=False):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.config = config
        self.use_focal_loss = use_focal_loss
        self.model = None
        self.history = None
        self.best_params = None
        
    def build_model(self, filters=[64, 128, 256], kernel_sizes=[3, 5, 7],
                   dropout_rate=0.3, dense_units=[256, 128], learning_rate=0.0001):
        from tensorflow.keras.regularizers import l2
        
        inputs = layers.Input(shape=(self.input_dim, 1))
        
        x = layers.LayerNormalization()(inputs)
        
        conv_outputs = []
        for kernel_size in kernel_sizes:
            branch = x
            
            for i, filter_size in enumerate(filters):
                branch = layers.Conv1D(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=None,
                    kernel_regularizer=l2(0.0001),
                    kernel_initializer='he_normal'
                )(branch)
                branch = layers.BatchNormalization()(branch)
                branch = layers.Activation('relu')(branch)
                
                if i == 0:
                    branch = layers.MaxPooling1D(pool_size=2)(branch)
            
            branch = layers.GlobalMaxPooling1D()(branch)
            branch = layers.Dropout(dropout_rate * 0.5)(branch)
            conv_outputs.append(branch)
        
        if len(conv_outputs) > 1:
            x = layers.Concatenate()(conv_outputs)
        else:
            x = conv_outputs[0]
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        for units in dense_units:
            x = layers.Dense(units, activation=None, kernel_regularizer=l2(0.0001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(self.n_classes, activation='softmax', 
                              kernel_initializer='glorot_uniform')(x)
        
        model = Model(inputs, outputs, name='improved_1d_cnn')
        
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            epsilon=1e-7
        )
        
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
             batch_size=64, epochs=100, patience=25, class_weight=None, verbose=1):
        print(f"\nTraining Improved 1D-CNN Classifier...")
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                mode='min',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if X_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
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
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = self.model.predict(X_reshaped, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X_reshaped, verbose=0)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, 
                                class_weight=None, n_trials=20):
        print("\n" + "="*80)
        print("OPTIMIZING IMPROVED CNN HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            import tensorflow as tf
            
            filters = trial.suggest_categorical(
                'filters',
                ['[64, 128, 256]', '[128, 256, 512]', '[64, 128, 256, 512]']
            )
            filters = eval(filters)
            
            kernel_sizes = trial.suggest_categorical(
                'kernel_sizes',
                ['[3, 5, 7]', '[3, 5]', '[5, 7, 9]']
            )
            kernel_sizes = eval(kernel_sizes)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            
            dense_units = trial.suggest_categorical(
                'dense_units',
                ['[256, 128]', '[512, 256]', '[512, 256, 128]']
            )
            dense_units = eval(dense_units)
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            self.build_model(
                filters=filters,
                kernel_sizes=kernel_sizes,
                dropout_rate=dropout_rate,
                dense_units=dense_units,
                learning_rate=learning_rate
            )
            
            history = self.train(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size,
                epochs=50,
                patience=15,
                class_weight=class_weight,
                verbose=0
            )
            
            val_loss = min(history.history['val_loss'])
            
            tf.keras.backend.clear_session()
            
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_params['filters'] = eval(self.best_params['filters'])
        self.best_params['kernel_sizes'] = eval(self.best_params['kernel_sizes'])
        self.best_params['dense_units'] = eval(self.best_params['dense_units'])
        
        print(f"\nBest CNN hyperparameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        
        self.build_model(
            filters=self.best_params['filters'],
            kernel_sizes=self.best_params['kernel_sizes'],
            dropout_rate=self.best_params['dropout_rate'],
            dense_units=self.best_params['dense_units'],
            learning_rate=self.best_params['learning_rate']
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
        
        self.model.save(save_dir / 'cnn_model.h5')
        
        if self.best_params is not None:
            joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        if self.history is not None:
            joblib.dump(self.history.history, save_dir / 'history.pkl')
        
        print(f"CNN model saved to {save_dir}")
    
    def load(self, save_dir):
        save_dir = Path(save_dir)
        
        self.model = keras.models.load_model(
            save_dir / 'cnn_model.h5',
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
        
        print(f"CNN model loaded from {save_dir}")
