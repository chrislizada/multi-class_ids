"""
Improved Denoising Autoencoder (DAE) Module
Supports multiple architectures and optimization strategies
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from pathlib import Path
import joblib


class DenoisingAutoencoder:
    def __init__(self, input_dim, config):
        self.input_dim = input_dim
        self.config = config
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None
        self.best_params = None
        
    def add_noise(self, data, noise_factor=0.2):
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
        noisy_data = data + noise
        return noisy_data
    
    def build_encoder(self, input_dim, encoder_layers, latent_dim, dropout_rate=0.3):
        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        
        for units in encoder_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
        
        latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
        
        encoder = Model(inputs, latent, name='encoder')
        return encoder
    
    def build_decoder(self, latent_dim, decoder_layers, output_dim, dropout_rate=0.3):
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = latent_inputs
        
        for units in decoder_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(output_dim, activation='linear')(x)
        
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    def build_autoencoder(self, encoder_layers=[512, 256], latent_dim=64, 
                         dropout_rate=0.3, learning_rate=0.001):
        decoder_layers = encoder_layers[::-1]
        
        self.encoder = self.build_encoder(
            self.input_dim, encoder_layers, latent_dim, dropout_rate
        )
        
        self.decoder = self.build_decoder(
            latent_dim, decoder_layers, self.input_dim, dropout_rate
        )
        
        inputs = layers.Input(shape=(self.input_dim,))
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        
        self.autoencoder = Model(inputs, outputs, name='denoising_autoencoder')
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def train(self, X_train, X_val=None, noise_factor=0.2, 
              batch_size=128, epochs=50, patience=10, verbose=1):
        print(f"\nTraining DAE with latent_dim={self.encoder.output_shape[-1]}...")
        
        X_train_noisy = self.add_noise(X_train, noise_factor)
        
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
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if X_val is not None:
            X_val_noisy = self.add_noise(X_val, noise_factor)
            validation_data = (X_val_noisy, X_val)
        else:
            validation_data = None
        
        self.history = self.autoencoder.fit(
            X_train_noisy, X_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def encode(self, X, batch_size=10000):
        # Process in batches to avoid OOM
        return self.encoder.predict(X, verbose=0, batch_size=batch_size)
    
    def decode(self, latent):
        return self.decoder.predict(latent, verbose=0)
    
    def reconstruct(self, X):
        return self.autoencoder.predict(X, verbose=0)
    
    def calculate_reconstruction_error(self, X):
        X_reconstructed = self.reconstruct(X)
        mse = np.mean((X - X_reconstructed) ** 2, axis=1)
        return mse
    
    def optimize_hyperparameters(self, X_train, X_val, n_trials=20):
        print("\n" + "="*80)
        print("OPTIMIZING DAE HYPERPARAMETERS")
        print("="*80 + "\n")
        
        def objective(trial):
            latent_dim = trial.suggest_categorical('latent_dim', self.config['latent_dims'])
            encoder_layers = trial.suggest_categorical(
                'encoder_layers', 
                [str(layers) for layers in self.config['encoder_layers']]
            )
            encoder_layers = eval(encoder_layers)
            
            noise_factor = trial.suggest_categorical('noise_factor', self.config['noise_factor'])
            dropout_rate = trial.suggest_categorical('dropout_rate', self.config['dropout_rate'])
            learning_rate = trial.suggest_categorical('learning_rate', self.config['learning_rate'])
            batch_size = trial.suggest_categorical('batch_size', self.config['batch_size'])
            
            self.build_autoencoder(
                encoder_layers=encoder_layers,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            history = self.train(
                X_train, X_val,
                noise_factor=noise_factor,
                batch_size=batch_size,
                epochs=self.config['epochs'],
                patience=self.config['patience'],
                verbose=0
            )
            
            val_loss = min(history.history['val_loss'])
            
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nBest hyperparameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  Best validation loss: {study.best_value:.6f}")
        
        encoder_layers = eval(self.best_params['encoder_layers'])
        
        self.build_autoencoder(
            encoder_layers=encoder_layers,
            latent_dim=self.best_params['latent_dim'],
            dropout_rate=self.best_params['dropout_rate'],
            learning_rate=self.best_params['learning_rate']
        )
        
        self.train(
            X_train, X_val,
            noise_factor=self.best_params['noise_factor'],
            batch_size=self.best_params['batch_size'],
            epochs=self.config['epochs'],
            patience=self.config['patience'],
            verbose=1
        )
        
        return self.best_params
    
    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder.save(save_dir / 'encoder.h5')
        self.decoder.save(save_dir / 'decoder.h5')
        self.autoencoder.save(save_dir / 'autoencoder.h5')
        
        if self.best_params is not None:
            joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        if self.history is not None:
            joblib.dump(self.history.history, save_dir / 'history.pkl')
        
        print(f"DAE model saved to {save_dir}")
    
    def load(self, save_dir):
        save_dir = Path(save_dir)
        
        self.encoder = keras.models.load_model(save_dir / 'encoder.h5')
        self.decoder = keras.models.load_model(save_dir / 'decoder.h5')
        self.autoencoder = keras.models.load_model(save_dir / 'autoencoder.h5')
        
        if (save_dir / 'best_params.pkl').exists():
            self.best_params = joblib.load(save_dir / 'best_params.pkl')
        
        if (save_dir / 'history.pkl').exists():
            history_dict = joblib.load(save_dir / 'history.pkl')
            
            class HistoryWrapper:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            self.history = HistoryWrapper(history_dict)
        
        print(f"DAE model loaded from {save_dir}")
