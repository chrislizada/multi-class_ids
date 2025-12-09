"""
Data preprocessing module with advanced feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.ohe = None
        self.dropped_constant_features = []
        self.dropped_correlated_features = []
        
    def load_data(self, file_path):
        print(f"Loading data from {file_path}...")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        print("Handling missing values...")
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', 
                             inplace=True)
        
        return df
    
    def handle_infinite_values(self, df):
        print("Handling infinite and extreme values...")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            # Replace infinity with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
            
            # Cap extreme values at 99.9th percentile
            if df[col].std() > 0:
                upper_limit = df[col].quantile(0.999)
                lower_limit = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        print("Infinite and extreme values handled")
        return df
    
    def identify_column_types(self, df, label_column='label'):
        df_features = df.drop(columns=[label_column], errors='ignore')
        
        self.categorical_columns = df_features.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        
        self.numerical_columns = df_features.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"Numerical columns: {len(self.numerical_columns)}")
        
        return self.categorical_columns, self.numerical_columns
    
    def encode_categorical_features(self, df, fit=True):
        print("Encoding categorical features...")
        
        from sklearn.preprocessing import OneHotEncoder
        
        if len(self.categorical_columns) == 0:
            return df
        
        # Convert all categorical columns to string to avoid mixed type errors
        for col in self.categorical_columns:
            df[col] = df[col].astype(str)
        
        if fit:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.ohe.fit_transform(df[self.categorical_columns])
            encoded_columns = self.ohe.get_feature_names_out(self.categorical_columns)
        else:
            if self.ohe is None:
                raise ValueError("OneHotEncoder not fitted. Call with fit=True first.")
            encoded = self.ohe.transform(df[self.categorical_columns])
            encoded_columns = self.ohe.get_feature_names_out(self.categorical_columns)
        
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
        
        df = df.drop(columns=self.categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def create_statistical_features(self, df):
        print("Creating statistical features...")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
        
        new_features = {}
        
        packet_length_cols = [col for col in numeric_cols if 'length' in col.lower() or 'len' in col.lower()]
        if len(packet_length_cols) >= 2:
            new_features['packet_length_std'] = df[packet_length_cols].std(axis=1)
            new_features['packet_length_range'] = df[packet_length_cols].max(axis=1) - df[packet_length_cols].min(axis=1)
            new_features['packet_length_cv'] = df[packet_length_cols].std(axis=1) / (df[packet_length_cols].mean(axis=1) + 1e-10)
        
        iat_cols = [col for col in numeric_cols if 'iat' in col.lower()]
        if len(iat_cols) >= 2:
            new_features['iat_std'] = df[iat_cols].std(axis=1)
            new_features['iat_range'] = df[iat_cols].max(axis=1) - df[iat_cols].min(axis=1)
        
        rate_cols = [col for col in numeric_cols if 'rate' in col.lower() or 'per_sec' in col.lower()]
        if len(rate_cols) >= 2:
            new_features['rate_mean'] = df[rate_cols].mean(axis=1)
            new_features['rate_max'] = df[rate_cols].max(axis=1)
        
        for col in numeric_cols[:10]:
            try:
                new_features[f'{col}_squared'] = df[col] ** 2
                new_features[f'{col}_log'] = np.log1p(np.abs(df[col]))
            except:
                continue
        
        if len(new_features) > 0:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)
            print(f"Created {len(new_features)} new statistical features")
        
        return df
    
    def remove_constant_features(self, df, threshold=0.99, fit=True):
        print("Removing constant and quasi-constant features...")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
        
        if fit:
            to_drop = []
            for col in numeric_cols:
                if df[col].nunique() == 1:
                    to_drop.append(col)
                elif df[col].value_counts(normalize=True).iloc[0] > threshold:
                    to_drop.append(col)
            
            self.dropped_constant_features = to_drop
        else:
            to_drop = [col for col in self.dropped_constant_features if col in df.columns]
        
        if len(to_drop) > 0:
            df = df.drop(columns=to_drop)
            print(f"Removed {len(to_drop)} constant/quasi-constant features")
        
        return df
    
    def remove_correlated_features(self, df, threshold=0.95, fit=True):
        print("Removing highly correlated features...")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
        
        if fit:
            corr_matrix = df[numeric_cols].corr().abs()
            
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            self.dropped_correlated_features = to_drop
        else:
            to_drop = [col for col in self.dropped_correlated_features if col in df.columns]
        
        if len(to_drop) > 0:
            df = df.drop(columns=to_drop)
            print(f"Removed {len(to_drop)} highly correlated features")
        
        return df
    
    def scale_features(self, df, label_column='label', fit=True):
        print("Scaling numerical features...")
        
        if label_column in df.columns:
            labels = df[label_column]
            df = df.drop(columns=[label_column])
        else:
            labels = None
        
        if fit:
            self.scaler = RobustScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_data = self.scaler.transform(df)
        
        df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        if labels is not None:
            df_scaled[label_column] = labels
        
        self.feature_names = df.columns.tolist()
        
        return df_scaled
    
    def encode_labels(self, labels, fit=True):
        print("Encoding labels...")
        
        if fit:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            print(f"Label classes: {self.label_encoder.classes_}")
        else:
            if self.label_encoder is None:
                raise ValueError("Label encoder not fitted. Call with fit=True first.")
            encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels
    
    def preprocess(self, df, label_column='label', fit=True):
        print("\n" + "="*80)
        print("STARTING DATA PREPROCESSING")
        print("="*80 + "\n")
        
        df = self.handle_missing_values(df)
        df = self.handle_infinite_values(df)
        
        if label_column in df.columns:
            labels = df[label_column]
            df_features = df.drop(columns=[label_column])
        else:
            raise ValueError(f"Label column '{label_column}' not found in dataframe")
        
        # Drop high-cardinality identifier columns that shouldn't be encoded
        id_columns = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 
                      'Timestamp', 'Flow_ID', 'Source_IP', 'Destination_IP',
                      'Source_Port', 'Destination_Port', 'stream',
                      'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
        
        columns_to_drop = [col for col in id_columns if col in df_features.columns]
        if columns_to_drop:
            print(f"Dropping identifier columns: {columns_to_drop}")
            df_features = df_features.drop(columns=columns_to_drop)
        
        # Identify column types FIRST
        self.identify_column_types(df_features)
        
        # Drop high-cardinality categorical columns (>100 unique values)
        # These would explode memory when one-hot encoded
        categorical_to_drop = []
        for col in self.categorical_columns:
            if col in df_features.columns:
                n_unique = df_features[col].nunique()
                if n_unique > 100:
                    print(f"Dropping high-cardinality categorical column: {col} ({n_unique:,} unique values)")
                    categorical_to_drop.append(col)
        
        if categorical_to_drop:
            df_features = df_features.drop(columns=categorical_to_drop)
            self.categorical_columns = [c for c in self.categorical_columns if c not in categorical_to_drop]
            print(f"Remaining categorical columns: {len(self.categorical_columns)}")
        
        df_features = self.encode_categorical_features(df_features, fit=fit)
        
        df_features = self.create_statistical_features(df_features)
        
        df_features = self.remove_constant_features(df_features, fit=fit)
        df_features = self.remove_correlated_features(df_features, fit=fit)
        
        df_features[label_column] = labels
        
        df_scaled = self.scale_features(df_features, label_column=label_column, fit=fit)
        
        labels_encoded = self.encode_labels(df_scaled[label_column], fit=fit)
        
        X = df_scaled.drop(columns=[label_column]).values
        y = labels_encoded
        
        print(f"\nPreprocessing complete!")
        print(f"Final feature shape: {X.shape}")
        print(f"Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = self.label_encoder.inverse_transform([label])[0]
            print(f"  {label_name}: {count} ({count/len(y)*100:.2f}%)")
        
        print("\n" + "="*80 + "\n")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        print("Splitting data into train/validation/test sets...")
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_train_val
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
