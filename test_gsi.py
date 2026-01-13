# Create test_gsi.py
from utils.data_generator import CarbonDataGenerator
from scoring.gsi_calculator import GreenScoreCalculator

# Generate data
generator = CarbonDataGenerator()
df = generator.generate_dataset(n_samples=1000)

# Initialize GSI
gsi_calc = GreenScoreCalculator()
gsi_calc.fit_population_benchmarks(df)

# Calculate GSI for recent data
df_with_gsi = gsi_calc.calculate_historical_gsi(df)

print(f"GSI Statistics:")
print(df_with_gsi['gsi_score'].describe())
print(f"\nGSI Ratings:")
print(df_with_gsi['gsi_rating'].value_counts())