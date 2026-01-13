# Create test_features.py
from utils.data_generator import CarbonDataGenerator
from features.feature_engineer import FeatureEngineer

# Generate data
generator = CarbonDataGenerator()
df = generator.generate_dataset(n_samples=100)

# Engineer features
engineer = FeatureEngineer()
df_eng, feature_cols = engineer.engineer_features(df, fit=True)

print(f"Original features: {df.shape[1]}")
print(f"Engineered features: {len(feature_cols)}")
print(f"Final shape: {df_eng.shape}")