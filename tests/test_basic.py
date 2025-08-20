"""
Basic tests for BandMatch modules
"""

import sys
import numpy as np
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bands import BandDefinition, FrequencyBand
from spectrum import SpectrumAnalyzer
from comparison import BandComparator, JudgmentLevel, ThresholdSet
from reference import ReferenceCombiner


def test_band_definition():
    """Test band definition creation and operations"""
    # Test default bands
    bands = BandDefinition()
    assert len(bands.bands) == 5
    assert bands.bands[0].name == "Low"
    assert bands.bands[0].low_freq == 20
    assert bands.bands[0].high_freq == 80
    
    # Test custom bands from string
    custom = BandDefinition.from_string("20-100,100-1000,1000-10000")
    assert len(custom.bands) == 3
    assert custom.bands[1].low_freq == 100
    assert custom.bands[1].high_freq == 1000
    
    # Test band names
    names = bands.get_band_names()
    assert "Low" in names
    assert "High" in names
    
    # Test frequency lookup
    band = bands.get_band_for_frequency(500)
    assert band is not None
    assert band.name == "Mid"


def test_fft_bin_mapping():
    """Test FFT bin calculation for bands"""
    bands = BandDefinition()
    n_fft = 4096
    sample_rate = 48000
    
    bin_mapping = bands.get_fft_bins(n_fft, sample_rate)
    
    # Check all bands have mappings
    for band in bands.bands:
        assert band.name in bin_mapping
        start, end = bin_mapping[band.name]
        assert start < end
        assert start >= 0
        assert end <= n_fft // 2 + 1


def test_spectrum_analyzer():
    """Test spectrum analysis on synthetic signal"""
    # Create test signal (1 second of 1000 Hz sine wave)
    sample_rate = 48000
    duration = 1.0
    frequency = 1000
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Analyze
    analyzer = SpectrumAnalyzer(n_fft=4096)
    band_db = analyzer.analyze_audio(signal, sample_rate)
    
    # Check that mid band (containing 1000 Hz) has highest energy
    assert "Mid" in band_db
    mid_energy = band_db["Mid"]
    
    # Mid should have higher energy than low and high
    assert mid_energy > band_db["Low"]
    assert mid_energy > band_db["High"]


def test_reference_combiner():
    """Test reference combination logic"""
    ref_a = {
        "Low": -20.0,
        "Mid": -15.0,
        "High": -18.0
    }
    
    ref_b = {
        "Low": -22.0,
        "Mid": -14.0,
        "High": -19.0
    }
    
    combiner = ReferenceCombiner(mismatch_threshold_db=3.0)
    
    # Test equal weight combination
    baseline = combiner.combine_references(ref_a, ref_b, weights=(1, 1))
    
    assert baseline["Low"] == -21.0  # Average of -20 and -22
    assert baseline["Mid"] == -14.5  # Average of -15 and -14
    assert baseline["High"] == -18.5  # Average of -18 and -19
    
    # Test weighted combination
    baseline_weighted = combiner.combine_references(ref_a, ref_b, weights=(2, 1))
    assert abs(baseline_weighted["Low"] - (-20.67)) < 0.1  # Weighted toward ref_a
    
    # Check warnings (none should be generated for small differences)
    warnings = combiner.get_warnings()
    assert len(warnings) == 0  # All differences < 3 dB


def test_band_comparator():
    """Test band comparison and judgment"""
    baseline = {
        "Low": -20.0,
        "Mid": -15.0,
        "High": -18.0
    }
    
    target = {
        "Low": -19.5,  # +0.5 dB (optimal)
        "Mid": -11.0,  # +4.0 dB (high)
        "High": -25.0   # -7.0 dB (very low)
    }
    
    comparator = BandComparator()
    comparisons = comparator.compare_bands(baseline, target)
    
    assert len(comparisons) == 3
    
    # Check judgments
    low_comp = next(c for c in comparisons if c.band_name == "Low")
    assert low_comp.judgment == JudgmentLevel.OPTIMAL
    assert abs(low_comp.delta_db - 0.5) < 0.01
    
    mid_comp = next(c for c in comparisons if c.band_name == "Mid")
    assert mid_comp.judgment == JudgmentLevel.HIGH
    assert abs(mid_comp.delta_db - 4.0) < 0.01
    
    high_comp = next(c for c in comparisons if c.band_name == "High")
    assert high_comp.judgment == JudgmentLevel.VERY_LOW
    assert abs(high_comp.delta_db - (-7.0)) < 0.01
    
    # Check match score
    score = comparator.calculate_overall_match_score(comparisons)
    assert 0 <= score <= 100


def test_threshold_judgment():
    """Test threshold-based judgment system"""
    threshold = ThresholdSet(optimal=1.0, slight=3.0, moderate=6.0)
    
    # Test various delta values
    assert threshold.judge(0.5) == JudgmentLevel.OPTIMAL
    assert threshold.judge(-0.5) == JudgmentLevel.OPTIMAL
    assert threshold.judge(2.0) == JudgmentLevel.SLIGHTLY_HIGH
    assert threshold.judge(-2.0) == JudgmentLevel.SLIGHTLY_LOW
    assert threshold.judge(4.0) == JudgmentLevel.HIGH
    assert threshold.judge(-4.0) == JudgmentLevel.LOW
    assert threshold.judge(7.0) == JudgmentLevel.VERY_HIGH
    assert threshold.judge(-7.0) == JudgmentLevel.VERY_LOW


def test_pink_noise_spectrum():
    """Test spectrum analysis with pink noise"""
    # Generate pink noise (1/f spectrum)
    sample_rate = 48000
    duration = 2.0
    samples = int(sample_rate * duration)
    
    # Simple pink noise generation
    white = np.random.randn(samples)
    # Apply simple 1/f filter (approximation)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1/sample_rate)
    freqs[0] = 1  # Avoid division by zero
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, len(white))
    
    # Normalize
    pink = pink / np.max(np.abs(pink))
    
    # Analyze
    analyzer = SpectrumAnalyzer(n_fft=4096)
    band_db = analyzer.analyze_audio(pink, sample_rate)
    
    # Pink noise should have decreasing energy with frequency
    # (approximately -3 dB per octave)
    assert band_db["Low"] > band_db["Mid"]
    assert band_db["Mid"] > band_db["High"]


if __name__ == "__main__":
    # Run tests
    print("Running BandMatch tests...")
    
    test_band_definition()
    print("✓ Band definition tests passed")
    
    test_fft_bin_mapping()
    print("✓ FFT bin mapping tests passed")
    
    test_spectrum_analyzer()
    print("✓ Spectrum analyzer tests passed")
    
    test_reference_combiner()
    print("✓ Reference combiner tests passed")
    
    test_band_comparator()
    print("✓ Band comparator tests passed")
    
    test_threshold_judgment()
    print("✓ Threshold judgment tests passed")
    
    test_pink_noise_spectrum()
    print("✓ Pink noise spectrum tests passed")
    
    print("\nAll tests passed! ✨")