"""Main training script for Carbon Footprint Forecaster."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.config import config
from utils.data_generator import CarbonDataGenerator
from features.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridCarbonForecaster
from scoring.gsi_calculator import GreenScoreCalculator
from recommendations.recommendation_engine import EcoRecommendationEngine

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("AI-DRIVEN HYBRID CARBON FOOTPRINT FORECASTER")
    print("="*60 + "\n")
    
    # Step 1: Generate data
    print("STEP 1: Generating synthetic data...")
    generator = CarbonDataGenerator(seed=42)
    n_samples = config.get('data.synthetic_samples', 10000)
    df = generator.generate_dataset(n_samples=n_samples)
    df = generator.add_user_profiles(df, n_users=100)
    df.to_csv('data/raw/carbon_footprint_data.csv')
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Saved to: data/raw/carbon_footprint_data.csv")
    
    # Step 2: Feature engineering
    print("\nSTEP 2: Feature engineering...")
    engineer = FeatureEngineer(config.get('features'))
    df_eng, feature_cols = engineer.engineer_features(df, fit=True)
    df_eng.to_csv('data/processed/engineered_features.csv')
    print(f"✓ Original features: {df.shape[1]}")
    print(f"✓ Engineered features: {len(feature_cols)}")
    print(f"✓ Saved to: data/processed/engineered_features.csv")
    
    # Step 3: Prepare data
    print("\nSTEP 3: Preparing train/val/test split...")
    X = df_eng[feature_cols].values
    y = df_eng['total_emission_kg_co2e'].values
    
    test_size = config.get('data.test_split', 0.2)
    val_size = config.get('data.validation_split', 0.1)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=42, shuffle=False)
    
    print(f"✓ Train: {X_train.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Test: {X_test.shape}")
    
    # Step 4: Train model
    print("\nSTEP 4: Training hybrid model...")
    model = HybridCarbonForecaster(
        num_features=X_train.shape[1],
        tcn_channels=config.get('model.tcn.num_channels'),
        attention_hidden=config.get('model.attention.hidden_dim'),
        attention_heads=config.get('model.attention.num_heads'),
        xgb_params=config.get('model.xgboost'),
        sequence_length=config.get('data.sequence_length'),
        use_tcn=True,
        use_attention=True,
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True)
    model.save('models/saved/hybrid_forecaster.pkl')
    print(f"✓ Model saved to: models/saved/hybrid_forecaster.pkl")
    
    # Step 5: Evaluate
    print("\nSTEP 5: Model evaluation...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Save metrics
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    pd.DataFrame([metrics]).to_csv('results/reports/metrics.csv', index=False)
    
    # Step 6: GSI Calculation
    print("\nSTEP 6: Calculating Green Score Index...")
    gsi_calc = GreenScoreCalculator(config.get('gsi'))
    gsi_calc.fit_population_benchmarks(df)
    df_with_gsi = gsi_calc.calculate_historical_gsi(df)
    df_with_gsi.to_csv('data/processed/data_with_gsi.csv')
    print(f"✓ GSI scores calculated")
    print(f"✓ Mean GSI: {df_with_gsi['gsi_score'].mean():.2f}")
    
    # Step 7: Generate sample recommendations
    print("\nSTEP 7: Generating recommendations...")
    rec_engine = EcoRecommendationEngine(config.get('recommendations'))
    
    sample_emissions = {
        'electricity_emission': df['electricity_emission'].median(),
        'vehicle_emission': df['vehicle_emission'].median(),
        'waste_emission': df['waste_emission'].median(),
        'water_emission': df['water_emission'].median(),
        'diet_emission': df['diet_emission'].median(),
    }
    
    recommendations = rec_engine.generate_recommendations(sample_emissions)
    print(f"✓ Generated {len(recommendations)} recommendations")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run dashboard: streamlit run app/dashboard.py")
    print("2. Check results in: results/")
    print("3. Review plots in: results/plots/")
    print("\n")

if __name__ == "__main__":
    main()