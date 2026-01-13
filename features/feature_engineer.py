"""Advanced feature engineering for carbon footprint prediction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List

class FeatureEngineer:
    """Extract and engineer features for carbon footprint forecasting."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input dataframe with datetime index
        
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        # Additional temporal indicators
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Season encoding
        df['season'] = df.index.month % 12 // 3
        df['season_spring'] = (df['season'] == 0).astype(int)
        df['season_summer'] = (df['season'] == 1).astype(int)
        df['season_fall'] = (df['season'] == 2).astype(int)
        df['season_winter'] = (df['season'] == 3).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_col: str,
                           lags: List[int] = None) -> pd.DataFrame:
        """Create lagged features for time series.
        
        Args:
            df: Input dataframe
            target_col: Column to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if lags is None:
            lags = [1, 7, 14, 30]
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               target_col: str,
                               windows: List[int] = None) -> pd.DataFrame:
        """Create rolling window statistics.
        
        Args:
            df: Input dataframe
            target_col: Column to create rolling features for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if windows is None:
            windows = [7, 14, 30]
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).mean())
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).std())
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).min())
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).max())
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """Create interaction features between variables.
        
        Args:
            df: Input dataframe
            feature_pairs: List of feature pairs to interact
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        if feature_pairs is None:
            # Default interactions
            feature_pairs = [
                ('electricity_kwh', 'temperature_factor'),
                ('vehicle_km', 'is_weekend'),
                ('waste_kg', 'recycling_rate'),
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame,
                                   cols: List[str]) -> pd.DataFrame:
        """Create statistical features across multiple columns.
        
        Args:
            df: Input dataframe
            cols: Columns to compute statistics for
        
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        available_cols = [col for col in cols if col in df.columns]
        
        if available_cols:
            df['total_energy'] = df[available_cols].sum(axis=1)
            df['mean_energy'] = df[available_cols].mean(axis=1)
            df['std_energy'] = df[available_cols].std(axis=1)
            df['max_energy'] = df[available_cols].max(axis=1)
            df['min_energy'] = df[available_cols].min(axis=1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                    categorical_cols: List[str] = None) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns
        
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen labels
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0]
                        if str(x) in self.label_encoders[col].classes_
                        else -1
                    )
        
        return df
    
    def create_emission_decomposition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose total emissions into components.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with decomposed emission features
        """
        df = df.copy()
        
        emission_cols = [col for col in df.columns if '_emission' in col and col != 'total_emission_kg_co2e']
        
        if 'total_emission_kg_co2e' in df.columns and emission_cols:
            total = df['total_emission_kg_co2e']
            for col in emission_cols:
                df[f'{col}_ratio'] = df[col] / (total + 1e-6)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         target_col: str = 'total_emission_kg_co2e',
                         fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Apply all feature engineering steps.
        
        Args:
            df: Input dataframe
            target_col: Target variable column name
            fit: Whether to fit scalers and encoders
        
        Returns:
            Tuple of (engineered dataframe, feature names)
        """
        df_eng = df.copy()
        
        # Temporal features
        df_eng = self.create_temporal_features(df_eng)
        
        # Lag features
        if target_col in df_eng.columns:
            df_eng = self.create_lag_features(df_eng, target_col)
            df_eng = self.create_rolling_features(df_eng, target_col)
        
        # Interaction features
        df_eng = self.create_interaction_features(df_eng)
        
        # Statistical features
        energy_cols = ['electricity_kwh', 'natural_gas_m3', 'vehicle_km']
        df_eng = self.create_statistical_features(df_eng, energy_cols)
        
        # Categorical encoding
        df_eng = self.encode_categorical_features(df_eng)
        
        # Emission decomposition
        df_eng = self.create_emission_decomposition(df_eng)
        
        # Drop rows with NaN values created by lag/rolling features
        df_eng = df_eng.dropna()
        
        # Get feature columns (exclude target and original categoricals)
        exclude_cols = [target_col, 'user_id'] + \
                      df.select_dtypes(include=['object', 'category']).columns.tolist()
        feature_cols = [col for col in df_eng.columns if col not in exclude_cols]
        
        # Scale features
        if fit:
            df_eng[feature_cols] = self.scaler.fit_transform(df_eng[feature_cols])
        else:
            df_eng[feature_cols] = self.scaler.transform(df_eng[feature_cols])
        
        self.feature_names = feature_cols
        
        return df_eng, feature_cols