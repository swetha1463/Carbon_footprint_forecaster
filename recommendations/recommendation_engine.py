"""AI-powered eco-behavior recommendation engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Recommendation:
    """Data class for eco-behavior recommendations."""
    category: str
    action: str
    description: str
    estimated_impact: float  # kg CO2e saved per day
    difficulty: str  # 'easy', 'medium', 'hard'
    priority_score: float
    timeframe: str  # 'immediate', 'short-term', 'long-term'

class EcoRecommendationEngine:
    """Context-aware recommendation engine for carbon reduction."""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_impact_threshold = self.config.get('min_impact_threshold', 5.0)
        self.max_recommendations = self.config.get('max_recommendations', 5)
        
        # Recommendation database
        self.recommendation_db = self._initialize_recommendations()
        
        # User history for personalization
        self.user_history = {}
    
    def _initialize_recommendations(self) -> List[Dict]:
        """Initialize database of possible recommendations."""
        return [
            # Electricity recommendations
            {
                'category': 'electricity',
                'action': 'Switch to LED bulbs',
                'description': 'Replace all incandescent bulbs with LED alternatives',
                'base_impact': 1.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('electricity_emission', 0) > 10,
            },
            {
                'category': 'electricity',
                'action': 'Install smart thermostat',
                'description': 'Use programmable thermostat to optimize heating/cooling',
                'base_impact': 3.0,
                'difficulty': 'medium',
                'timeframe': 'short-term',
                'conditions': lambda e: e.get('electricity_emission', 0) > 15,
            },
            {
                'category': 'electricity',
                'action': 'Unplug vampire devices',
                'description': 'Disconnect devices when not in use to eliminate standby power',
                'base_impact': 0.8,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
            {
                'category': 'electricity',
                'action': 'Switch to renewable energy',
                'description': 'Contact energy provider about green energy plans',
                'base_impact': 8.0,
                'difficulty': 'medium',
                'timeframe': 'short-term',
                'conditions': lambda e: e.get('electricity_emission', 0) > 12,
            },
            {
                'category': 'electricity',
                'action': 'Install solar panels',
                'description': 'Invest in rooftop solar panels for clean energy generation',
                'base_impact': 15.0,
                'difficulty': 'hard',
                'timeframe': 'long-term',
                'conditions': lambda e: e.get('electricity_emission', 0) > 20,
            },
            
            # Transportation recommendations
            {
                'category': 'transportation',
                'action': 'Use public transportation',
                'description': 'Take bus/train for commute instead of personal vehicle',
                'base_impact': 4.0,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 5,
            },
            {
                'category': 'transportation',
                'action': 'Carpool to work',
                'description': 'Share rides with colleagues to reduce individual emissions',
                'base_impact': 3.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 5,
            },
            {
                'category': 'transportation',
                'action': 'Bike or walk for short trips',
                'description': 'Use active transportation for distances under 5km',
                'base_impact': 2.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 3,
            },
            {
                'category': 'transportation',
                'action': 'Switch to hybrid/electric vehicle',
                'description': 'Consider upgrading to a more efficient vehicle',
                'base_impact': 10.0,
                'difficulty': 'hard',
                'timeframe': 'long-term',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 8,
            },
            {
                'category': 'transportation',
                'action': 'Optimize driving habits',
                'description': 'Practice eco-driving: smooth acceleration, maintain speed',
                'base_impact': 1.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 4,
            },
            {
                'category': 'transportation',
                'action': 'Work from home',
                'description': 'Request remote work options to eliminate commute emissions',
                'base_impact': 5.0,
                'difficulty': 'medium',
                'timeframe': 'short-term',
                'conditions': lambda e: e.get('vehicle_emission', 0) > 6,
            },
            
            # Waste recommendations
            {
                'category': 'waste',
                'action': 'Increase recycling rate',
                'description': 'Properly sort and recycle paper, plastic, glass, and metal',
                'base_impact': 0.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('waste_emission', 0) > 0.5,
            },
            {
                'category': 'waste',
                'action': 'Start composting',
                'description': 'Compost organic waste to reduce landfill emissions',
                'base_impact': 0.8,
                'difficulty': 'medium',
                'timeframe': 'short-term',
                'conditions': lambda e: e.get('waste_emission', 0) > 0.7,
            },
            {
                'category': 'waste',
                'action': 'Reduce single-use plastics',
                'description': 'Use reusable bags, bottles, and containers',
                'base_impact': 0.4,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
            {
                'category': 'waste',
                'action': 'Buy products with minimal packaging',
                'description': 'Choose products with recyclable or minimal packaging',
                'base_impact': 0.3,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
            
            # Water recommendations
            {
                'category': 'water',
                'action': 'Install low-flow fixtures',
                'description': 'Replace showerheads and faucets with water-efficient models',
                'base_impact': 0.2,
                'difficulty': 'medium',
                'timeframe': 'short-term',
                'conditions': lambda e: e.get('water_emission', 0) > 0.05,
            },
            {
                'category': 'water',
                'action': 'Fix leaks promptly',
                'description': 'Repair dripping faucets and running toilets',
                'base_impact': 0.15,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
            {
                'category': 'water',
                'action': 'Reduce shower time',
                'description': 'Limit showers to 5-7 minutes',
                'base_impact': 0.1,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('water_emission', 0) > 0.04,
            },
            
            # Diet recommendations
            {
                'category': 'diet',
                'action': 'Reduce meat consumption',
                'description': 'Try meatless meals 2-3 times per week',
                'base_impact': 2.0,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: e.get('diet_emission', 0) > 2.0,
            },
            {
                'category': 'diet',
                'action': 'Choose local and seasonal produce',
                'description': 'Buy from local farmers markets to reduce transportation emissions',
                'base_impact': 1.0,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
            {
                'category': 'diet',
                'action': 'Reduce food waste',
                'description': 'Plan meals, store food properly, and use leftovers',
                'base_impact': 1.5,
                'difficulty': 'easy',
                'timeframe': 'immediate',
                'conditions': lambda e: True,
            },
        ]
    
    def generate_recommendations(self, 
                                emissions: Dict[str, float],
                                user_profile: Dict = None,
                                gsi_components: Dict = None) -> List[Recommendation]:
        """Generate personalized recommendations based on user data.
        
        Args:
            emissions: Current emission values by category
            user_profile: User characteristics (optional)
            gsi_components: GSI score components (optional)
        
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        for rec_template in self.recommendation_db:
            # Check if conditions are met
            if not rec_template['conditions'](emissions):
                continue
            
            # Calculate actual impact based on user's emissions
            impact = self._calculate_impact(rec_template, emissions, user_profile)
            
            # Skip if impact is below threshold
            if impact < self.min_impact_threshold and rec_template['difficulty'] == 'hard':
                continue
            
            # Calculate priority score
            priority = self._calculate_priority(
                rec_template, impact, emissions, gsi_components, user_profile)
            
            rec = Recommendation(
                category=rec_template['category'],
                action=rec_template['action'],
                description=rec_template['description'],
                estimated_impact=impact,
                difficulty=rec_template['difficulty'],
                priority_score=priority,
                timeframe=rec_template['timeframe'],
            )
            
            recommendations.append(rec)
        
        # Sort by priority and return top N
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        return recommendations[:self.max_recommendations]
    
    def _calculate_impact(self, rec_template: Dict,
                         emissions: Dict[str, float],
                         user_profile: Dict = None) -> float:
        """Calculate actual impact for user.
        
        Args:
            rec_template: Recommendation template
            emissions: User emissions
            user_profile: User characteristics
        
        Returns:
            Estimated impact in kg CO2e
        """
        base_impact = rec_template['base_impact']
        category = rec_template['category']
        
        # Scale impact based on current emissions
        emission_key = f'{category}_emission'
        if emission_key in emissions:
            current_emission = emissions[emission_key]
            # Impact is proportional to current emissions, capped at base impact
            impact = min(base_impact, current_emission * 0.3)
        else:
            impact = base_impact * 0.5  # Default scaling
        
        # Adjust for user profile
        if user_profile:
            household_size = user_profile.get('household_size', 1)
            impact *= np.sqrt(household_size)  # Economies of scale
        
        return round(impact, 2)
    
    def _calculate_priority(self, rec_template: Dict,
                           impact: float,
                           emissions: Dict[str, float],
                           gsi_components: Dict = None,
                           user_profile: Dict = None) -> float:
        """Calculate priority score for recommendation.
        
        Args:
            rec_template: Recommendation template
            impact: Estimated impact
            emissions: User emissions
            gsi_components: GSI components
            user_profile: User profile
        
        Returns:
            Priority score
        """
        # Base score from impact (normalized)
        impact_score = min(100, impact * 5)
        
        # Difficulty penalty (easier = higher priority)
        difficulty_map = {'easy': 1.0, 'medium': 0.8, 'hard': 0.6}
        difficulty_multiplier = difficulty_map.get(rec_template['difficulty'], 0.7)
        
        # Timeframe preference (immediate actions prioritized)
        timeframe_map = {'immediate': 1.0, 'short-term': 0.85, 'long-term': 0.7}
        timeframe_multiplier = timeframe_map.get(rec_template['timeframe'], 0.75)
        
        # Category weight (from GSI weights if available)
        category_weight = 1.0
        if gsi_components and 'category_scores' in gsi_components:
            category = rec_template['category']
            if category in gsi_components['category_scores']:
                # Lower score = higher priority
                category_score = gsi_components['category_scores'][category]
                category_weight = 1.5 - (category_score / 100)
        
        # Environmental awareness modifier
        awareness_multiplier = 1.0
        if user_profile and 'environmental_awareness' in user_profile:
            awareness = user_profile['environmental_awareness']
            # High awareness users get harder recommendations prioritized more
            if rec_template['difficulty'] == 'hard':
                awareness_multiplier = 0.8 + 0.4 * awareness
        
        # Calculate final priority
        priority = (impact_score * difficulty_multiplier * timeframe_multiplier * 
                   category_weight * awareness_multiplier)
        
        return round(priority, 2)
    
    def track_recommendation_adoption(self, user_id: str, 
                                     recommendation: Recommendation,
                                     adopted: bool):
        """Track whether user adopted a recommendation.
        
        Args:
            user_id: User identifier
            recommendation: Recommendation object
            adopted: Whether the recommendation was adopted
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append({
            'timestamp': pd.Timestamp.now(),
            'category': recommendation.category,
            'action': recommendation.action,
            'adopted': adopted,
            'estimated_impact': recommendation.estimated_impact,
        })
    
    def get_user_impact_summary(self, user_id: str) -> Dict:
        """Calculate total impact from adopted recommendations.
        
        Args:
            user_id: User identifier
        
        Returns:
            Summary of user's impact
        """
        if user_id not in self.user_history:
            return {'total_adopted': 0, 'estimated_total_impact': 0}
        
        history = self.user_history[user_id]
        adopted = [h for h in history if h['adopted']]
        
        return {
            'total_recommended': len(history),
            'total_adopted': len(adopted),
            'adoption_rate': len(adopted) / len(history) if history else 0,
            'estimated_total_impact': sum(h['estimated_impact'] for h in adopted),
            'by_category': self._summarize_by_category(adopted),
        }
    
    def _summarize_by_category(self, adopted: List[Dict]) -> Dict:
        """Summarize adopted recommendations by category.
        
        Args:
            adopted: List of adopted recommendations
        
        Returns:
            Summary by category
        """
        summary = {}
        
        for rec in adopted:
            category = rec['category']
            if category not in summary:
                summary[category] = {
                    'count': 0,
                    'total_impact': 0,
                }
            summary[category]['count'] += 1
            summary[category]['total_impact'] += rec['estimated_impact']
        
        return summary