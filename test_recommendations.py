# Create test_recommendations.py
from recommendations.recommendation_engine import EcoRecommendationEngine

# Sample emissions
emissions = {
    'electricity_emission': 12.5,
    'vehicle_emission': 8.3,
    'waste_emission': 1.2,
    'water_emission': 0.05,
    'diet_emission': 2.5,
}

# Generate recommendations
rec_engine = EcoRecommendationEngine()
recommendations = rec_engine.generate_recommendations(emissions)

print(f"Generated {len(recommendations)} recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.action}")
    print(f"   Impact: {rec.estimated_impact:.2f} kg CO2e/day")
    print(f"   Difficulty: {rec.difficulty}")