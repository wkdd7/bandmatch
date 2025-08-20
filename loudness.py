"""
Loudness measurement and normalization module for BandMatch
Implements ITU-R BS.1770-4 LUFS measurement and normalization
"""

import numpy as np
import pyloudnorm as pyln
from typing import Tuple, Optional, Union


class LoudnessProcessor:
    """LUFS measurement and normalization"""
    
    # Common LUFS targets for different applications
    LUFS_PRESETS = {
        'streaming': -14.0,      # Spotify, YouTube, Apple Music
        'broadcast': -23.0,      # EBU R128 broadcast standard
        'cd': -9.0,             # CD mastering level
        'podcast': -16.0,        # Podcast standard
        'film': -27.0,          # Film/cinema standard
        'custom': -14.0         # Default custom value
    }
    
    def __init__(self, target_lufs: float = -14.0):
        """
        Initialize loudness processor
        
        Args:
            target_lufs: Target integrated LUFS level
        """
        self.target_lufs = target_lufs
        self.meter = None
        self._sample_rate = None
    
    def set_sample_rate(self, sample_rate: int):
        """
        Set sample rate and initialize meter
        
        Args:
            sample_rate: Audio sample rate
        """
        if self._sample_rate != sample_rate:
            self._sample_rate = sample_rate
            self.meter = pyln.Meter(sample_rate)
    
    def measure_loudness(self, 
                        audio: np.ndarray, 
                        sample_rate: int) -> dict:
        """
        Measure various loudness metrics
        
        Args:
            audio: Audio data (mono or stereo)
            sample_rate: Sample rate
            
        Returns:
            Dictionary with loudness measurements
        """
        self.set_sample_rate(sample_rate)
        
        # Ensure audio is in correct shape for pyloudnorm
        if audio.ndim == 1:
            # Mono - add channel dimension
            audio_reshaped = audio.reshape(-1, 1)
        else:
            # Already has channel dimension
            audio_reshaped = audio
        
        # Measure integrated loudness
        integrated_lufs = self.meter.integrated_loudness(audio_reshaped)
        
        # Get loudness range
        loudness_range = self._calculate_loudness_range(audio_reshaped)
        
        # Get true peak
        true_peak_db = self._calculate_true_peak(audio_reshaped)
        
        # Get short-term loudness statistics
        short_term_stats = self._calculate_short_term_stats(audio_reshaped)
        
        return {
            'integrated_lufs': integrated_lufs,
            'loudness_range': loudness_range,
            'true_peak_db': true_peak_db,
            'short_term_max': short_term_stats['max'],
            'short_term_min': short_term_stats['min'],
            'short_term_mean': short_term_stats['mean']
        }
    
    def normalize_to_target(self, 
                           audio: np.ndarray, 
                           sample_rate: int,
                           target_lufs: Optional[float] = None,
                           peak_normalize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Normalize audio to target LUFS
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            target_lufs: Target LUFS (uses instance default if None)
            peak_normalize: Also apply peak normalization to prevent clipping
            
        Returns:
            Tuple of (normalized_audio, gain_applied_db)
        """
        if target_lufs is None:
            target_lufs = self.target_lufs
        
        self.set_sample_rate(sample_rate)
        
        # Measure current loudness
        if audio.ndim == 1:
            audio_reshaped = audio.reshape(-1, 1)
            was_mono = True
        else:
            audio_reshaped = audio
            was_mono = False
        
        current_lufs = self.meter.integrated_loudness(audio_reshaped)
        
        # Skip if already at inf (silence)
        if np.isinf(current_lufs):
            return audio, 0.0
        
        # Calculate required gain
        gain_db = target_lufs - current_lufs
        
        # Apply normalization
        normalized = pyln.normalize.loudness(
            audio_reshaped, 
            current_lufs, 
            target_lufs
        )
        
        # Apply peak normalization if requested
        if peak_normalize:
            peak = np.max(np.abs(normalized))
            if peak > 1.0:
                normalized = normalized / peak
                gain_db = gain_db - 20 * np.log10(peak)
        
        # Restore original shape if was mono
        if was_mono:
            normalized = normalized.squeeze()
        
        return normalized, gain_db
    
    def calculate_gain_to_target(self,
                                 current_lufs: float,
                                 target_lufs: Optional[float] = None) -> float:
        """
        Calculate gain needed to reach target LUFS
        
        Args:
            current_lufs: Current integrated LUFS
            target_lufs: Target LUFS
            
        Returns:
            Required gain in dB
        """
        if target_lufs is None:
            target_lufs = self.target_lufs
        
        if np.isinf(current_lufs):
            return 0.0
        
        return target_lufs - current_lufs
    
    def apply_gain(self, 
                   audio: np.ndarray, 
                   gain_db: float) -> np.ndarray:
        """
        Apply gain to audio
        
        Args:
            audio: Audio data
            gain_db: Gain to apply in dB
            
        Returns:
            Audio with gain applied
        """
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def match_loudness(self,
                      reference_audio: np.ndarray,
                      target_audio: np.ndarray,
                      sample_rate: int) -> Tuple[np.ndarray, float]:
        """
        Match target audio loudness to reference
        
        Args:
            reference_audio: Reference audio
            target_audio: Audio to match
            sample_rate: Sample rate
            
        Returns:
            Tuple of (matched_audio, gain_applied_db)
        """
        self.set_sample_rate(sample_rate)
        
        # Measure reference loudness
        if reference_audio.ndim == 1:
            ref_reshaped = reference_audio.reshape(-1, 1)
        else:
            ref_reshaped = reference_audio
        
        ref_lufs = self.meter.integrated_loudness(ref_reshaped)
        
        # Normalize target to reference level
        return self.normalize_to_target(target_audio, sample_rate, ref_lufs)
    
    def _calculate_loudness_range(self, audio: np.ndarray) -> float:
        """
        Calculate loudness range (LRA)
        
        Args:
            audio: Audio data with shape (samples, channels)
            
        Returns:
            Loudness range in LU
        """
        try:
            # This is a simplified version - full LRA calculation is complex
            # For now, return a placeholder or use momentary loudness variance
            return 0.0  # Placeholder
        except Exception:
            return 0.0
    
    def _calculate_true_peak(self, audio: np.ndarray) -> float:
        """
        Calculate true peak level
        
        Args:
            audio: Audio data
            
        Returns:
            True peak in dBFS
        """
        peak = np.max(np.abs(audio))
        if peak == 0:
            return -np.inf
        return 20 * np.log10(peak)
    
    def _calculate_short_term_stats(self, audio: np.ndarray) -> dict:
        """
        Calculate short-term loudness statistics
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with min, max, mean short-term loudness
        """
        # Simplified version - actual implementation would use 
        # 3-second sliding window
        return {
            'min': self.meter.integrated_loudness(audio),
            'max': self.meter.integrated_loudness(audio),
            'mean': self.meter.integrated_loudness(audio)
        }
    
    @staticmethod
    def db_to_linear(db: float) -> float:
        """Convert dB to linear scale"""
        return 10 ** (db / 20)
    
    @staticmethod
    def linear_to_db(linear: float) -> float:
        """Convert linear to dB scale"""
        if linear == 0:
            return -np.inf
        return 20 * np.log10(linear)


def batch_normalize_loudness(audio_list: list,
                            sample_rate: int,
                            target_lufs: float = -14.0,
                            use_average: bool = False) -> list:
    """
    Normalize multiple audio files to same loudness
    
    Args:
        audio_list: List of audio arrays
        sample_rate: Sample rate
        target_lufs: Target LUFS level
        use_average: If True, normalize all to their average LUFS
        
    Returns:
        List of (normalized_audio, gain_db) tuples
    """
    processor = LoudnessProcessor(target_lufs)
    
    if use_average:
        # Measure all and calculate average
        lufs_values = []
        for audio in audio_list:
            metrics = processor.measure_loudness(audio, sample_rate)
            if not np.isinf(metrics['integrated_lufs']):
                lufs_values.append(metrics['integrated_lufs'])
        
        if lufs_values:
            target_lufs = np.mean(lufs_values)
    
    # Normalize all to target
    results = []
    for audio in audio_list:
        normalized, gain = processor.normalize_to_target(
            audio, sample_rate, target_lufs
        )
        results.append((normalized, gain))
    
    return results


def validate_loudness_matching(audio_list: list,
                              sample_rate: int,
                              tolerance_lu: float = 1.0) -> Tuple[bool, str]:
    """
    Validate that audio files are matched in loudness
    
    Args:
        audio_list: List of audio arrays
        sample_rate: Sample rate
        tolerance_lu: Tolerance in loudness units
        
    Returns:
        Tuple of (is_matched, message)
    """
    processor = LoudnessProcessor()
    
    lufs_values = []
    for i, audio in enumerate(audio_list):
        metrics = processor.measure_loudness(audio, sample_rate)
        lufs_values.append(metrics['integrated_lufs'])
    
    # Check if all within tolerance
    lufs_range = np.ptp([l for l in lufs_values if not np.isinf(l)])
    
    if lufs_range <= tolerance_lu:
        return True, f"Loudness matched within {lufs_range:.1f} LU"
    else:
        return False, f"Loudness mismatch: {lufs_range:.1f} LU (tolerance: {tolerance_lu} LU)"