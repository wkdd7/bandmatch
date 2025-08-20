"""
Band comparison and judgment module for BandMatch
Compares target against baseline and provides judgments
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class JudgmentLevel(Enum):
    """Judgment levels for band differences"""
    OPTIMAL = "적정"
    SLIGHTLY_LOW = "약간 부족"
    SLIGHTLY_HIGH = "약간 과다"
    LOW = "부족"
    HIGH = "과다"
    VERY_LOW = "크게 부족"
    VERY_HIGH = "크게 과다"


@dataclass
class BandComparison:
    """Result of comparing a single band"""
    band_name: str
    baseline_db: float
    target_db: float
    delta_db: float
    judgment: JudgmentLevel
    eq_suggestion: str
    confidence: float  # 0-1 confidence in the judgment
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'band_name': self.band_name,
            'baseline_db': round(self.baseline_db, 1),
            'target_db': round(self.target_db, 1),
            'delta_db': round(self.delta_db, 1),
            'judgment': self.judgment.value,
            'eq_suggestion': self.eq_suggestion,
            'confidence': round(self.confidence, 2)
        }


class ThresholdSet:
    """Set of thresholds for judgments"""
    
    def __init__(self,
                 optimal: float = 1.0,
                 slight: float = 3.0,
                 moderate: float = 6.0):
        """
        Initialize threshold set
        
        Args:
            optimal: Threshold for optimal (±dB)
            slight: Threshold for slight deviation (±dB)
            moderate: Threshold for moderate deviation (±dB)
        """
        self.optimal = optimal
        self.slight = slight
        self.moderate = moderate
    
    def judge(self, delta_db: float) -> JudgmentLevel:
        """
        Judge based on delta
        
        Args:
            delta_db: Difference in dB
            
        Returns:
            JudgmentLevel
        """
        abs_delta = abs(delta_db)
        
        if abs_delta < self.optimal:
            return JudgmentLevel.OPTIMAL
        elif abs_delta < self.slight:
            if delta_db > 0:
                return JudgmentLevel.SLIGHTLY_HIGH
            else:
                return JudgmentLevel.SLIGHTLY_LOW
        elif abs_delta < self.moderate:
            if delta_db > 0:
                return JudgmentLevel.HIGH
            else:
                return JudgmentLevel.LOW
        else:
            if delta_db > 0:
                return JudgmentLevel.VERY_HIGH
            else:
                return JudgmentLevel.VERY_LOW


class BandComparator:
    """Compares target against baseline and provides judgments"""
    
    # Default thresholds for different band types
    DEFAULT_THRESHOLDS = {
        'default': ThresholdSet(1.0, 3.0, 6.0),
        'low': ThresholdSet(1.5, 3.5, 7.0),      # More tolerance for low frequencies
        'high': ThresholdSet(0.8, 2.5, 5.0)      # Less tolerance for high frequencies
    }
    
    def __init__(self, 
                 thresholds: Optional[Dict[str, ThresholdSet]] = None):
        """
        Initialize comparator
        
        Args:
            thresholds: Custom thresholds per band
        """
        self.thresholds = thresholds or {}
        self.default_threshold = self.DEFAULT_THRESHOLDS['default']
    
    def compare_bands(self,
                     baseline_db: Dict[str, float],
                     target_db: Dict[str, float]) -> List[BandComparison]:
        """
        Compare target against baseline
        
        Args:
            baseline_db: Baseline band dB values
            target_db: Target band dB values
            
        Returns:
            List of BandComparison results
        """
        comparisons = []
        
        for band_name in baseline_db.keys():
            if band_name not in target_db:
                continue
            
            baseline_val = baseline_db[band_name]
            target_val = target_db[band_name]
            delta = target_val - baseline_val
            
            # Get appropriate threshold
            threshold = self._get_threshold_for_band(band_name)
            
            # Make judgment
            judgment = threshold.judge(delta)
            
            # Generate EQ suggestion
            eq_suggestion = self._generate_eq_suggestion(band_name, delta, judgment)
            
            # Calculate confidence
            confidence = self._calculate_confidence(delta, threshold)
            
            comparison = BandComparison(
                band_name=band_name,
                baseline_db=baseline_val,
                target_db=target_val,
                delta_db=delta,
                judgment=judgment,
                eq_suggestion=eq_suggestion,
                confidence=confidence
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _get_threshold_for_band(self, band_name: str) -> ThresholdSet:
        """Get threshold set for a specific band"""
        if band_name in self.thresholds:
            return self.thresholds[band_name]
        
        # Use category-based defaults
        band_lower = band_name.lower()
        if 'low' in band_lower or 'sub' in band_lower or 'bass' in band_lower:
            return self.DEFAULT_THRESHOLDS.get('low', self.default_threshold)
        elif 'high' in band_lower or 'presence' in band_lower or 'air' in band_lower:
            return self.DEFAULT_THRESHOLDS.get('high', self.default_threshold)
        else:
            return self.default_threshold
    
    def _generate_eq_suggestion(self, 
                               band_name: str, 
                               delta_db: float,
                               judgment: JudgmentLevel) -> str:
        """
        Generate EQ adjustment suggestion
        
        Args:
            band_name: Band name
            delta_db: Difference in dB
            judgment: Judgment level
            
        Returns:
            EQ suggestion string
        """
        if judgment == JudgmentLevel.OPTIMAL:
            return "No adjustment needed"
        
        # Get frequency range for band (simplified)
        freq_ranges = {
            'Low': '20-80 Hz',
            'Low-Mid': '80-250 Hz',
            'Mid': '250-2k Hz',
            'High-Mid': '2-6 kHz',
            'High': '6-20 kHz'
        }
        
        freq_range = freq_ranges.get(band_name, band_name)
        adjustment = abs(delta_db)
        
        if delta_db > 0:
            # Target is higher than baseline - need to reduce
            action = "reduce"
            direction = f"-{adjustment:.1f} dB"
        else:
            # Target is lower than baseline - need to boost
            action = "boost"
            direction = f"+{adjustment:.1f} dB"
        
        # Generate suggestion based on severity
        if judgment in [JudgmentLevel.SLIGHTLY_LOW, JudgmentLevel.SLIGHTLY_HIGH]:
            return f"Consider slight {action} around {freq_range} ({direction})"
        elif judgment in [JudgmentLevel.LOW, JudgmentLevel.HIGH]:
            return f"Recommend {action} at {freq_range} ({direction})"
        else:  # VERY_LOW or VERY_HIGH
            return f"Strongly recommend {action} at {freq_range} ({direction})"
    
    def _calculate_confidence(self, delta_db: float, threshold: ThresholdSet) -> float:
        """
        Calculate confidence in judgment
        
        Args:
            delta_db: Difference in dB
            threshold: Threshold set used
            
        Returns:
            Confidence value (0-1)
        """
        abs_delta = abs(delta_db)
        
        # High confidence near threshold boundaries
        if abs_delta < threshold.optimal * 0.5:
            return 1.0  # Very clearly optimal
        elif abs_delta > threshold.moderate * 1.5:
            return 1.0  # Very clearly extreme
        else:
            # Lower confidence near boundaries
            distances = [
                abs(abs_delta - threshold.optimal),
                abs(abs_delta - threshold.slight),
                abs(abs_delta - threshold.moderate)
            ]
            min_distance = min(distances)
            # Confidence decreases near boundaries
            confidence = 1.0 - (min_distance / threshold.optimal) * 0.3
            return max(0.5, min(1.0, confidence))
    
    def generate_overall_summary(self, 
                                comparisons: List[BandComparison]) -> str:
        """
        Generate overall summary text
        
        Args:
            comparisons: List of band comparisons
            
        Returns:
            Summary text
        """
        # Count judgments
        judgment_counts = {}
        for comp in comparisons:
            judgment = comp.judgment
            judgment_counts[judgment] = judgment_counts.get(judgment, 0) + 1
        
        # Find most significant issues
        issues = []
        for comp in comparisons:
            if abs(comp.delta_db) >= 3.0:
                if comp.delta_db > 0:
                    issues.append(f"{comp.band_name} +{comp.delta_db:.1f} dB")
                else:
                    issues.append(f"{comp.band_name} {comp.delta_db:.1f} dB")
        
        # Build summary
        if not issues:
            return "Overall tonal balance is well-matched to references"
        
        summary_parts = []
        
        # Add main issues
        if len(issues) <= 2:
            summary_parts.append(f"Main differences: {', '.join(issues)}")
        else:
            summary_parts.append(f"Multiple band differences detected")
        
        # Add character assessment
        high_bands_high = sum(1 for c in comparisons 
                             if 'High' in c.band_name and c.delta_db > 3)
        low_bands_high = sum(1 for c in comparisons 
                            if 'Low' in c.band_name and c.delta_db > 3)
        
        if high_bands_high > low_bands_high:
            summary_parts.append("Target is brighter than references")
        elif low_bands_high > high_bands_high:
            summary_parts.append("Target is warmer/darker than references")
        
        return ". ".join(summary_parts)
    
    def calculate_overall_match_score(self, 
                                     comparisons: List[BandComparison]) -> float:
        """
        Calculate overall match score (0-100)
        
        Args:
            comparisons: List of band comparisons
            
        Returns:
            Match score
        """
        if not comparisons:
            return 0.0
        
        # Weight by confidence and delta
        scores = []
        for comp in comparisons:
            # Convert delta to score (0 dB = 100, 10 dB = 0)
            delta_score = max(0, 100 * (1 - abs(comp.delta_db) / 10))
            # Weight by confidence
            weighted_score = delta_score * comp.confidence
            scores.append(weighted_score)
        
        return np.mean(scores)


class EQRecommendation:
    """Generates EQ recommendations based on comparisons"""
    
    @staticmethod
    def generate_eq_curve(comparisons: List[BandComparison],
                         sample_rate: int = 48000) -> Dict:
        """
        Generate EQ curve recommendations
        
        Args:
            comparisons: List of band comparisons
            sample_rate: Sample rate for frequency points
            
        Returns:
            Dictionary with EQ curve data
        """
        eq_points = []
        
        for comp in comparisons:
            if comp.judgment == JudgmentLevel.OPTIMAL:
                continue
            
            # Get band center frequency (simplified)
            center_freqs = {
                'Low': 50,
                'Low-Mid': 165,
                'Mid': 1000,
                'High-Mid': 4000,
                'High': 10000
            }
            
            freq = center_freqs.get(comp.band_name, 1000)
            gain = -comp.delta_db  # Inverse to correct
            
            eq_points.append({
                'frequency': freq,
                'gain_db': round(gain, 1),
                'q': 0.7,  # Default Q factor
                'type': 'bell'
            })
        
        return {
            'eq_points': eq_points,
            'sample_rate': sample_rate,
            'recommended': len(eq_points) > 0
        }
    
    @staticmethod
    def prioritize_adjustments(comparisons: List[BandComparison],
                              max_adjustments: int = 3) -> List[BandComparison]:
        """
        Prioritize which adjustments to make first
        
        Args:
            comparisons: List of band comparisons
            max_adjustments: Maximum number of adjustments to recommend
            
        Returns:
            Prioritized list of comparisons needing adjustment
        """
        # Filter out optimal bands
        needs_adjustment = [c for c in comparisons 
                          if c.judgment != JudgmentLevel.OPTIMAL]
        
        # Sort by absolute delta (biggest problems first)
        needs_adjustment.sort(key=lambda x: abs(x.delta_db), reverse=True)
        
        # Return top N
        return needs_adjustment[:max_adjustments]