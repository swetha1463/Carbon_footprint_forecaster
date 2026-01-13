# Create test_generator.py
from utils.data_generator import CarbonDataGenerator

generator = CarbonDataGenerator()
df = generator.generate_dataset(n_samples=100)
print(f"Generated {len(df)} samples")
print(df.head())
print(f"Columns: {df.columns.tolist()}")