"""Interactive demo script."""
from utils.data_generator import CarbonDataGenerator
from features.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridCarbonForecaster
from scoring.gsi_calculator import GreenScoreCalculator
from recommendations.recommendation_engine import EcoRecommendationEngine
import pandas as pd

def run_demo():
    """Run interactive demonstration."""
    print("\n" + "="*70)
    print("  🌍 CARBON FOOTPRINT FORECASTER - INTERACTIVE DEMO")
    print("="*70 + "\n")
    
    # User input
    print("📝 Enter your daily activities:\n")
    
    electricity = float(input("Electricity consumption (kWh): ") or "15")
    vehicle_km = float(input("Vehicle distance (km): ") or "30")
    waste = float(input("Waste generated (kg): ") or "2")
    water = float(input("Water usage (liters): ") or "150")
    
    diet = input("Diet type (omnivore/vegetarian/vegan): ") or "omnivore"
    
    # Calculate emissions
    print("\n⚙️  Calculating your carbon footprint...\n")
    
    emission_factors = {
        'electricity': 0.92,
        'vehicle_petrol': 2.31,
        'waste': 0.5,
        'water': 0.0003,
    }
    
    diet_factors = {'omnivore': 2.5, 'vegetarian': 1.7, 'vegan': 1.5}
    
    electricity_emission = electricity * emission_factors['electricity']
    vehicle_emission = (vehicle_km * 7.5 / 100) * emission_factors['vehicle_petrol']
    waste_emission = waste * 0.6 * emission_factors['waste']
    water_emission = water * emission_factors['water']
    diet_emission = diet_factors.get(diet, 2.5)
    
    total_emission = (electricity_emission + vehicle_emission + 
                     waste_emission + water_emission + diet_emission)
    
    # Display results
    print("="*70)
    print("📊 YOUR CARBON FOOTPRINT RESULTS")
    print("="*70)
    print(f"\n🔥 Total Daily Emissions: {total_emission:.2f} kg CO₂e\n")
    print("Breakdown:")
    print(f"  • Electricity: {electricity_emission:.2f} kg CO₂e ({electricity_emission/total_emission*100:.1f}%)")
    print(f"  • Transportation: {vehicle_emission:.2f} kg CO₂e ({vehicle_emission/total_emission*100:.1f}%)")
    print(f"  • Waste: {waste_emission:.2f} kg CO₂e ({waste_emission/total_emission*100:.1f}%)")
    print(f"  • Water: {water_emission:.2f} kg CO₂e ({water_emission/total_emission*100:.1f}%)")
    print(f"  • Diet: {diet_emission:.2f} kg CO₂e ({diet_emission/total_emission*100:.1f}%)")
    
    print(f"\n📅 Monthly Projection: {total_emission * 30:.1f} kg CO₂e")
    print(f"📆 Annual Projection: {total_emission * 365 / 1000:.2f} tons CO₂e")
    
    # Generate recommendations
    print("\n" + "="*70)
    print("💡 PERSONALIZED RECOMMENDATIONS")
    print("="*70 + "\n")
    
    rec_engine = EcoRecommendationEngine()
    emissions_dict = {
        'electricity_emission': electricity_emission,
        'vehicle_emission': vehicle_emission,
        'waste_emission': waste_emission,
        'water_emission': water_emission,
        'diet_emission': diet_emission,
    }
    
    recommendations = rec_engine.generate_recommendations(emissions_dict)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.action}")
        print(f"   Impact: Save {rec.estimated_impact:.2f} kg CO₂e/day")
        print(f"   Difficulty: {rec.difficulty.upper()}")
        print(f"   Timeframe: {rec.timeframe}")
        print()
    
    print("="*70)
    print("\n Demo complete! Launch dashboard: streamlit run app/dashboard.py")
    print()

if __name__ == "__main__":
    run_demo()