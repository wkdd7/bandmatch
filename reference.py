"""
Reference combination and mismatch detection module for BandMatch
Handles averaging of reference tracks and consistency checking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReferenceWarning:
    """Warning about reference mismatch"""
    band_name: str
    difference_db: float
    ref_a_db: float
    ref_b_db: float
    severity: str  # 'low', 'medium', 'high'
    
    def __str__(self):
        return f"Reference disagreement at {self.band_name}: {self.difference_db:.1f} dB"


class ReferenceCombiner:
    """Combines multiple reference tracks into baseline"""
    
    def __init__(self, mismatch_threshold_db: float = 3.0):
        """
        Initialize reference combiner
        
        Args:
            mismatch_threshold_db: Threshold for mismatch warning
        """
        self.mismatch_threshold_db = mismatch_threshold_db
        self.warnings = []
    
    def combine_references(self,
                          ref_a_db: Dict[str, float],
                          ref_b_db: Dict[str, float],
                          weights: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, float]:
        """
        Combine two references into baseline
        
        Args:
            ref_a_db: Band dB values for reference A
            ref_b_db: Band dB values for reference B
            weights: Weight tuple for weighted average
            
        Returns:
            Combined baseline dB values
        """
        # Normalize weights
        total_weight = sum(weights)
        w_a = weights[0] / total_weight
        w_b = weights[1] / total_weight
        
        baseline = {}
        self.warnings = []
        
        # Check bands match
        if set(ref_a_db.keys()) != set(ref_b_db.keys()):
            raise ValueError("Reference tracks have different band definitions")
        
        for band_name in ref_a_db.keys():
            a_db = ref_a_db[band_name]
            b_db = ref_b_db[band_name]
            
            # Weighted average
            baseline[band_name] = w_a * a_db + w_b * b_db
            
            # Check for mismatch
            diff = abs(a_db - b_db)
            if diff >= self.mismatch_threshold_db:
                severity = self._get_severity(diff)
                warning = ReferenceWarning(
                    band_name=band_name,
                    difference_db=diff,
                    ref_a_db=a_db,
                    ref_b_db=b_db,
                    severity=severity
                )
                self.warnings.append(warning)
        
        return baseline
    
    def combine_multiple_references(self,
                                   references: List[Dict[str, float]],
                                   weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Combine multiple references into baseline
        
        Args:
            references: List of band dB dictionaries
            weights: Optional weights for each reference
            
        Returns:
            Combined baseline dB values
        """
        if len(references) == 0:
            raise ValueError("No references provided")
        
        if len(references) == 1:
            return references[0].copy()
        
        # Default equal weights
        if weights is None:
            weights = [1.0] * len(references)
        
        if len(weights) != len(references):
            raise ValueError("Number of weights must match number of references")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Check all have same bands
        band_names = set(references[0].keys())
        for ref in references[1:]:
            if set(ref.keys()) != band_names:
                raise ValueError("All references must have same band definitions")
        
        # Weighted average
        baseline = {}
        self.warnings = []
        
        for band_name in band_names:
            values = [ref[band_name] for ref in references]
            baseline[band_name] = sum(v * w for v, w in zip(values, normalized_weights))
            
            # Check for mismatches (pairwise)
            self._check_pairwise_mismatches(band_name, values, references)
        
        return baseline
    
    def _check_pairwise_mismatches(self, 
                                   band_name: str, 
                                   values: List[float],
                                   references: List[Dict[str, float]]):
        """Check for mismatches between all pairs of references"""
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(values[i] - values[j])
                if diff >= self.mismatch_threshold_db:
                    severity = self._get_severity(diff)
                    warning = ReferenceWarning(
                        band_name=band_name,
                        difference_db=diff,
                        ref_a_db=values[i],
                        ref_b_db=values[j],
                        severity=severity
                    )
                    self.warnings.append(warning)
    
    def _get_severity(self, difference_db: float) -> str:
        """
        Determine severity of mismatch
        
        Args:
            difference_db: Absolute difference in dB
            
        Returns:
            Severity level
        """
        if difference_db < self.mismatch_threshold_db:
            return 'low'
        elif difference_db < self.mismatch_threshold_db * 2:
            return 'medium'
        else:
            return 'high'
    
    def get_warnings(self) -> List[ReferenceWarning]:
        """Get list of warnings from last combination"""
        return self.warnings
    
    def get_warning_summary(self) -> str:
        """Get summary of warnings"""
        if not self.warnings:
            return "References are well-matched"
        
        high_warnings = [w for w in self.warnings if w.severity == 'high']
        medium_warnings = [w for w in self.warnings if w.severity == 'medium']
        
        summary = []
        if high_warnings:
            summary.append(f"{len(high_warnings)} high severity mismatches")
        if medium_warnings:
            summary.append(f"{len(medium_warnings)} medium severity mismatches")
        
        return ", ".join(summary)
    
    def calculate_reference_similarity(self,
                                      ref_a_db: Dict[str, float],
                                      ref_b_db: Dict[str, float]) -> float:
        """
        Calculate overall similarity between two references
        
        Args:
            ref_a_db: Band dB values for reference A
            ref_b_db: Band dB values for reference B
            
        Returns:
            Similarity score (0-100)
        """
        differences = []
        for band_name in ref_a_db.keys():
            if band_name in ref_b_db:
                diff = abs(ref_a_db[band_name] - ref_b_db[band_name])
                differences.append(diff)
        
        if not differences:
            return 0.0
        
        # Calculate similarity based on average difference
        avg_diff = np.mean(differences)
        
        # Convert to 0-100 scale (0 dB diff = 100%, 10 dB diff = 0%)
        similarity = max(0, 100 * (1 - avg_diff / 10))
        
        return similarity
    
    def suggest_weight_adjustment(self,
                                 ref_a_db: Dict[str, float],
                                 ref_b_db: Dict[str, float],
                                 target_characteristic: str = 'balanced') -> Tuple[float, float]:
        """
        Suggest weight adjustment based on reference characteristics
        
        Args:
            ref_a_db: Band dB values for reference A
            ref_b_db: Band dB values for reference B
            target_characteristic: Target sound ('balanced', 'bright', 'warm')
            
        Returns:
            Suggested weights for (ref_a, ref_b)
        """
        # Calculate spectral characteristics
        a_brightness = self._calculate_brightness(ref_a_db)
        b_brightness = self._calculate_brightness(ref_b_db)
        
        if target_characteristic == 'balanced':
            # Equal weights for balanced
            return (1.0, 1.0)
        elif target_characteristic == 'bright':
            # Weight brighter reference more
            if a_brightness > b_brightness:
                return (1.5, 0.5)
            else:
                return (0.5, 1.5)
        elif target_characteristic == 'warm':
            # Weight warmer (less bright) reference more
            if a_brightness < b_brightness:
                return (1.5, 0.5)
            else:
                return (0.5, 1.5)
        else:
            return (1.0, 1.0)
    
    def _calculate_brightness(self, band_db: Dict[str, float]) -> float:
        """
        Calculate brightness score based on high frequency content
        
        Args:
            band_db: Band dB values
            
        Returns:
            Brightness score
        """
        # Simple heuristic: ratio of high to low frequencies
        high_bands = ['High-Mid', 'High', 'Presence', 'Air', 'Brilliance']
        low_bands = ['Low', 'Low-Mid', 'Bass', 'Sub', 'Sub-Bass', 'Fundamental']
        
        high_energy = sum(band_db.get(b, 0) for b in high_bands if b in band_db)
        low_energy = sum(band_db.get(b, 0) for b in low_bands if b in band_db)
        
        if low_energy == 0:
            return 100.0
        
        return high_energy - low_energy


class ReferenceValidator:
    """Validates reference tracks for suitability"""
    
    @staticmethod
    def validate_reference_quality(band_db: Dict[str, float],
                                  metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Validate if a track is suitable as reference
        
        Args:
            band_db: Band dB values
            metadata: Audio metadata
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check duration
        if metadata.get('duration', 0) < 30:
            issues.append(f"Short duration: {metadata.get('duration', 0):.1f}s (recommend > 30s)")
        
        # Check for extreme band values
        for band_name, db_value in band_db.items():
            if db_value < -60:
                issues.append(f"Very low energy in {band_name}: {db_value:.1f} dB")
            elif db_value > 0:
                issues.append(f"Clipping possible in {band_name}: {db_value:.1f} dB")
        
        # Check overall dynamic range
        db_values = list(band_db.values())
        db_range = max(db_values) - min(db_values)
        if db_range < 6:
            issues.append(f"Low spectral contrast: {db_range:.1f} dB")
        elif db_range > 40:
            issues.append(f"Extreme spectral contrast: {db_range:.1f} dB")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def check_genre_compatibility(ref_a_features: Dict,
                                 ref_b_features: Dict) -> Tuple[bool, str]:
        """
        Check if references are from compatible genres
        
        Args:
            ref_a_features: Features/characteristics of reference A
            ref_b_features: Features/characteristics of reference B
            
        Returns:
            Tuple of (are_compatible, message)
        """
        # This is a simplified check - could be expanded with ML classification
        
        # Check spectral centroid difference
        centroid_diff = abs(
            ref_a_features.get('spectral_centroid_mean', 0) - 
            ref_b_features.get('spectral_centroid_mean', 0)
        )
        
        if centroid_diff > 2000:  # Hz
            return False, f"Large spectral centroid difference: {centroid_diff:.0f} Hz"
        
        return True, "References appear compatible"