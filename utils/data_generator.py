"""Synthetic data generation for training and testing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

class CarbonDataGenerator:
    """Generate synthetic carbon footprint data with realistic patterns."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.emission_factors = {
            'electricity': 0.92,  # kg CO2e per kWh
            'natural_gas': 5.3,   # kg CO2e per m³
            'vehicle_petrol': 2.31,  # kg CO2e per liter
            'vehicle_diesel': 2.68,  # kg CO2e per liter
            'public_transport': 0.14,  # kg CO2e per km
            'waste': 0.5,  # kg CO2e per kg waste
            'water': 0.0003,  # kg CO2e per liter
        }
    
    def generate_dataset(self, n_samples: int = 10000, 
                        start_date: str = '2022-01-01') -> pd.DataFrame:
        """Generate comprehensive synthetic dataset.
        
        Args:
            n_samples: Number of samples to generate
            start_date: Starting date for the dataset
        
        Returns:
            DataFrame with synthetic carbon footprint data
        """
        dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            # Base patterns with seasonal variation
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend_factor = 0.8 if date.dayofweek >= 5 else 1.0
            
            # Electricity consumption (kWh) - higher in summer/winter
            electricity = max(0, np.random.normal(
                15 * season_factor * weekend_factor, 3))
            
            # Natural gas (m³) - higher in winter
            winter_factor = 1.5 if date.month in [12, 1, 2] else 0.5
            natural_gas = max(0, np.random.normal(
                5 * winter_factor * weekend_factor, 1))
            
            # Vehicle usage (km)
            commute_km = max(0, np.random.normal(
                30 * weekend_factor, 8)) if date.dayofweek < 5 else np.random.normal(10, 5)
            
            # Fuel type and consumption
            vehicle_type = np.random.choice(['petrol', 'diesel', 'electric'], 
                                           p=[0.6, 0.3, 0.1])
            fuel_efficiency = np.random.uniform(6, 10)  # L/100km or kWh/100km
            fuel_consumed = commute_km * fuel_efficiency / 100
            
            # Public transport (km)
            public_transport_km = max(0, np.random.normal(
                5 * weekend_factor, 2))
            
            # Waste generation (kg)
            waste = max(0, np.random.normal(
                2 * weekend_factor, 0.5))
            recycling_rate = np.random.uniform(0.2, 0.6)
            
            # Water usage (liters)
            water = max(0, np.random.normal(
                150 * weekend_factor, 30))
            
            # Dietary impact (simplified)
            diet_type = np.random.choice(['omnivore', 'vegetarian', 'vegan'],
                                        p=[0.7, 0.2, 0.1])
            diet_factor = {'omnivore': 2.5, 'vegetarian': 1.7, 'vegan': 1.5}
            
            # Calculate emissions
            electricity_emission = electricity * self.emission_factors['electricity']
            gas_emission = natural_gas * self.emission_factors['natural_gas']
            
            if vehicle_type == 'electric':
                vehicle_emission = fuel_consumed * self.emission_factors['electricity']
            else:
                vehicle_emission = fuel_consumed * self.emission_factors[f'vehicle_{vehicle_type}']
            
            transport_emission = public_transport_km * self.emission_factors['public_transport']
            waste_emission = waste * (1 - recycling_rate) * self.emission_factors['waste']
            water_emission = water * self.emission_factors['water']
            diet_emission = diet_factor[diet_type]
            
            total_emission = (electricity_emission + gas_emission + 
                            vehicle_emission + transport_emission + 
                            waste_emission + water_emission + diet_emission)
            
            # Add some random noise
            total_emission *= np.random.uniform(0.95, 1.05)
            
            data.append({
                'date': date,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'day_of_year': date.dayofyear,
                'is_weekend': date.dayofweek >= 5,
                'electricity_kwh': electricity,
                'natural_gas_m3': natural_gas,
                'vehicle_km': commute_km,
                'vehicle_type': vehicle_type,
                'fuel_consumed': fuel_consumed,
                'public_transport_km': public_transport_km,
                'waste_kg': waste,
                'recycling_rate': recycling_rate,
                'water_liters': water,
                'diet_type': diet_type,
                'electricity_emission': electricity_emission,
                'gas_emission': gas_emission,
                'vehicle_emission': vehicle_emission,
                'transport_emission': transport_emission,
                'waste_emission': waste_emission,
                'water_emission': water_emission,
                'diet_emission': diet_emission,
                'total_emission_kg_co2e': total_emission,
                'temperature_factor': season_factor,
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    def add_user_profiles(self, df: pd.DataFrame, 
                         n_users: int = 100) -> pd.DataFrame:
        """Add user profile information to dataset.
        
        Args:
            df: Base dataframe
            n_users: Number of unique users
        
        Returns:
            DataFrame with user profiles
        """
        df_with_users = df.copy()
        
        # Assign random user IDs
        df_with_users['user_id'] = np.random.randint(1, n_users + 1, len(df))
        
        # User characteristics
        user_profiles = {}
        for user_id in range(1, n_users + 1):
            user_profiles[user_id] = {
                'household_size': np.random.randint(1, 6),
                'home_size_sqm': np.random.uniform(50, 200),
                'income_level': np.random.choice(['low', 'medium', 'high'],
                                                 p=[0.3, 0.5, 0.2]),
                'environmental_awareness': np.random.uniform(0, 1),
            }
        
        df_with_users['household_size'] = df_with_users['user_id'].map(
            lambda x: user_profiles[x]['household_size'])
        df_with_users['home_size_sqm'] = df_with_users['user_id'].map(
            lambda x: user_profiles[x]['home_size_sqm'])
        df_with_users['income_level'] = df_with_users['user_id'].map(
            lambda x: user_profiles[x]['income_level'])
        df_with_users['environmental_awareness'] = df_with_users['user_id'].map(
            lambda x: user_profiles[x]['environmental_awareness'])
        
        return df_with_users
    
    def generate_realtime_sample(self, base_values: dict = None) -> dict:
        """Generate a single real-time data sample.
        
        Args:
            base_values: Base values to use for generation
        
        Returns:
            Dictionary with sample data
        """
        if base_values is None:
            base_values = {
                'electricity_kwh': 15,
                'vehicle_km': 30,
                'waste_kg': 2,
                'water_liters': 150,
            }
        
        # Add random variation
        sample = {}
        for key, value in base_values.items():
            sample[key] = max(0, value * np.random.uniform(0.8, 1.2))
        
        sample['timestamp'] = datetime.now()
        
        return sample