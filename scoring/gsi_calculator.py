"""Green Score Index (GSI) calculator with temporal dynamics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class GreenScoreCalculator:
    """Calculate dynamic Green Score Index (GSI) for users."""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: Configuration dictionary with weights and benchmarks
        """
        self.config = config or {}
        
        # Default category weights
        self.weights = self.config.get('weights', {
            'electricity': 0.3,
            'transportation': 0.35,
            'waste': 0.2,
            'water': 0.15,
        })
        
        # Benchmark percentiles for scoring
        self.benchmarks = self.config.get('benchmark_percentiles', {
            'excellent': 90,
            'good': 75,
            'average': 50,
            'poor': 25,
        })
        
        # Population statistics (will be updated from data)
        self.population_stats = {}
        
        # Temporal decay factor for momentum
        self.momentum_decay = 0.95
        
    def fit_population_benchmarks(self, df: pd.DataFrame,
                                  emission_columns: List[str] = None):
        """Calculate population-level benchmarks from historical data.
        
        Args:
            df: Historical emissions dataframe
            emission_columns: Columns containing emission values
        """
        if emission_columns is None:
            emission_columns = [col for col in df.columns if '_emission' in col]
        
        self.population_stats = {}
        
        for col in emission_columns:
            if col in df.columns:
                category = col.replace('_emission', '')
                self.population_stats[category] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'percentiles': {
                        p: df[col].quantile(p/100) 
                        for p in [10, 25, 50, 75, 90, 95, 99]
                    }
                }
        
        # Total emissions statistics
        if 'total_emission_kg_co2e' in df.columns:
            total_col = 'total_emission_kg_co2e'
            self.population_stats['total'] = {
                'mean': df[total_col].mean(),
                'std': df[total_col].std(),
                'percentiles': {
                    p: df[total_col].quantile(p/100)
                    for p in [10, 25, 50, 75, 90, 95, 99]
                }
            }
    
    def calculate_category_score(self, value: float, category: str) -> float:
        """Calculate score for a single category.
        
        Args:
            value: Emission value
            category: Category name
        
        Returns:
            Score between 0-100 (higher is better)
        """
        if category not in self.population_stats:
            return 50.0  # Default neutral score
        
        stats = self.population_stats[category]
        percentiles = stats['percentiles']
        
        # Find which percentile the value falls into
        if value <= percentiles[10]:
            score = 100  # Excellent
        elif value <= percentiles[25]:
            score = 90 - (value - percentiles[10]) / (percentiles[25] - percentiles[10]) * 10
        elif value <= percentiles[50]:
            score = 75 - (value - percentiles[25]) / (percentiles[50] - percentiles[25]) * 15
        elif value <= percentiles[75]:
            score = 50 - (value - percentiles[50]) / (percentiles[75] - percentiles[50]) * 25
        elif value <= percentiles[90]:
            score = 25 - (value - percentiles[75]) / (percentiles[90] - percentiles[75]) * 15
        else:
            # Beyond 90th percentile
            score = max(0, 10 - (value - percentiles[90]) / (percentiles[99] - percentiles[90]) * 10)
        
        return max(0, min(100, score))
    
    def calculate_gsi(self, emissions: Dict[str, float],
                     previous_gsi: float = None,
                     behavioral_streak: int = 0) -> Dict[str, float]:
        """Calculate comprehensive Green Score Index.
        
        Args:
            emissions: Dictionary of emission values by category
            previous_gsi: Previous GSI score for momentum calculation
            behavioral_streak: Number of consecutive days with improved behavior
        
        Returns:
            Dictionary with GSI components and final score
        """
        category_scores = {}
        
        # Calculate individual category scores
        for category, weight in self.weights.items():
            emission_key = f'{category}_emission'
            
            if emission_key in emissions:
                score = self.calculate_category_score(
                    emissions[emission_key], category)
                category_scores[category] = score
            else:
                category_scores[category] = 50.0  # Neutral default
        
        # Weighted average base score
        base_gsi = sum(
            category_scores[cat] * weight 
            for cat, weight in self.weights.items()
        )
        
        # Momentum bonus (rewards consistent improvement)
        momentum_bonus = 0
        if previous_gsi is not None:
            improvement = base_gsi - previous_gsi
            momentum_bonus = min(5, max(-5, improvement * 0.5))
        
        # Behavioral streak bonus (rewards consistency)
        streak_bonus = min(10, behavioral_streak * 0.5)
        
        # Final GSI with bonuses
        final_gsi = min(100, base_gsi + momentum_bonus + streak_bonus)
        
        return {
            'category_scores': category_scores,
            'base_gsi': base_gsi,
            'momentum_bonus': momentum_bonus,
            'streak_bonus': streak_bonus,
            'final_gsi': final_gsi,
            'rating': self._get_rating(final_gsi),
        }
    
    def _get_rating(self, gsi: float) -> str:
        """Convert GSI score to rating category.
        
        Args:
            gsi: GSI score
        
        Returns:
            Rating string
        """
        if gsi >= 85:
            return 'Excellent'
        elif gsi >= 70:
            return 'Good'
        elif gsi >= 50:
            return 'Average'
        elif gsi >= 30:
            return 'Below Average'
        else:
            return 'Needs Improvement'
    
    def calculate_historical_gsi(self, df: pd.DataFrame,
                                emission_cols: List[str] = None) -> pd.DataFrame:
        """Calculate GSI for historical data.
        
        Args:
            df: Historical emissions dataframe
            emission_cols: List of emission columns
        
        Returns:
            DataFrame with GSI scores added
        """
        df = df.copy()
        
        if emission_cols is None:
            emission_cols = [col for col in df.columns if '_emission' in col]
        
        gsi_scores = []
        previous_gsi = None
        streak = 0
        
        for idx, row in df.iterrows():
            emissions = {col: row[col] for col in emission_cols if col in row}
            
            gsi_result = self.calculate_gsi(emissions, previous_gsi, streak)
            
            # Update streak
            if previous_gsi is not None:
                if gsi_result['final_gsi'] > previous_gsi:
                    streak += 1
                else:
                    streak = 0
            
            gsi_scores.append(gsi_result['final_gsi'])
            previous_gsi = gsi_result['final_gsi']
        
        df['gsi_score'] = gsi_scores
        df['gsi_rating'] = df['gsi_score'].apply(self._get_rating)
        
        return df
    
    def get_improvement_potential(self, emissions: Dict[str, float]) -> Dict[str, Dict]:
        """Calculate potential for improvement in each category.
        
        Args:
            emissions: Current emission values
        
        Returns:
            Dictionary with improvement potential by category
        """
        improvements = {}
        
        for category, weight in self.weights.items():
            emission_key = f'{category}_emission'
            
            if emission_key not in emissions or category not in self.population_stats:
                continue
            
            current_value = emissions[emission_key]
            stats = self.population_stats[category]
            
            # Calculate potential if user reaches different benchmarks
            targets = {
                'excellent': stats['percentiles'][10],
                'good': stats['percentiles'][25],
                'average': stats['percentiles'][50],
            }
            
            improvements[category] = {
                'current': current_value,
                'current_score': self.calculate_category_score(current_value, category),
                'targets': targets,
                'potential_savings': {
                    level: max(0, current_value - target)
                    for level, target in targets.items()
                },
                'potential_scores': {
                    level: self.calculate_category_score(target, category)
                    for level, target in targets.items()
                },
                'weight': weight,
            }
        
        return improvements
    
    def compare_with_peers(self, emissions: Dict[str, float],
                          peer_group: str = 'all') -> Dict[str, any]:
        """Compare user's emissions with peer groups.
        
        Args:
            emissions: User's emission values
            peer_group: Peer group identifier
        
        Returns:
            Comparison statistics
        """
        comparisons = {}
        
        total_emission = sum(
            emissions.get(f'{cat}_emission', 0) 
            for cat in self.weights.keys()
        )
        
        if 'total' in self.population_stats:
            total_stats = self.population_stats['total']
            
            percentile_rank = sum(
                1 for p, val in total_stats['percentiles'].items()
                if total_emission <= val
            ) / len(total_stats['percentiles']) * 100
            
            comparisons['total'] = {
                'user_value': total_emission,
                'population_mean': total_stats['mean'],
                'population_median': total_stats['percentiles'][50],
                'percentile_rank': percentile_rank,
                'better_than_percent': 100 - percentile_rank,
            }
        
        return comparisons