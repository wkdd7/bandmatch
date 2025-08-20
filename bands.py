"""
Frequency band definition and FFT bin mapping module for BandMatch
Handles frequency band boundaries and bin calculations
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class FrequencyBand:
    """Represents a frequency band"""
    name: str
    low_freq: float
    high_freq: float
    color: str = "#808080"  # Default gray color for visualization
    
    def __str__(self):
        return f"{self.name}: {self.low_freq}-{self.high_freq} Hz"
    
    def contains(self, frequency: float) -> bool:
        """Check if frequency is within this band"""
        return self.low_freq <= frequency < self.high_freq


class BandDefinition:
    """Manages frequency band definitions and FFT bin mapping"""
    
    # Default band definitions
    DEFAULT_BANDS = [
        FrequencyBand("Low", 20, 80, "#FF6B6B"),        # Red
        FrequencyBand("Low-Mid", 80, 250, "#FFA500"),   # Orange
        FrequencyBand("Mid", 250, 2000, "#4ECDC4"),     # Teal
        FrequencyBand("High-Mid", 2000, 6000, "#45B7D1"), # Blue
        FrequencyBand("High", 6000, 20000, "#96CEB4")   # Green
    ]
    
    # Preset band configurations for different use cases
    PRESETS = {
        'default': DEFAULT_BANDS,
        'mastering': [
            FrequencyBand("Sub", 20, 60, "#8B0000"),
            FrequencyBand("Bass", 60, 200, "#FF6B6B"),
            FrequencyBand("Low-Mid", 200, 500, "#FFA500"),
            FrequencyBand("Mid", 500, 2000, "#4ECDC4"),
            FrequencyBand("Upper-Mid", 2000, 5000, "#45B7D1"),
            FrequencyBand("Presence", 5000, 10000, "#96CEB4"),
            FrequencyBand("Air", 10000, 20000, "#DDA0DD")
        ],
        'podcast': [
            FrequencyBand("Low", 50, 120, "#FF6B6B"),
            FrequencyBand("Low-Mid", 120, 350, "#FFA500"),
            FrequencyBand("Mid", 350, 2000, "#4ECDC4"),
            FrequencyBand("High-Mid", 2000, 5000, "#45B7D1"),
            FrequencyBand("High", 5000, 16000, "#96CEB4")
        ],
        'edm': [
            FrequencyBand("Sub-Bass", 20, 60, "#8B0000"),
            FrequencyBand("Bass", 60, 250, "#FF6B6B"),
            FrequencyBand("Low-Mid", 250, 500, "#FFA500"),
            FrequencyBand("Mid", 500, 2000, "#4ECDC4"),
            FrequencyBand("High-Mid", 2000, 8000, "#45B7D1"),
            FrequencyBand("High", 8000, 20000, "#96CEB4")
        ],
        'voice': [
            FrequencyBand("Fundamental", 80, 250, "#FF6B6B"),
            FrequencyBand("Lower-Harmonics", 250, 500, "#FFA500"),
            FrequencyBand("Upper-Harmonics", 500, 2000, "#4ECDC4"),
            FrequencyBand("Presence", 2000, 4000, "#45B7D1"),
            FrequencyBand("Brilliance", 4000, 12000, "#96CEB4")
        ]
    }
    
    def __init__(self, bands: Optional[List[FrequencyBand]] = None, 
                 preset: str = 'default'):
        """
        Initialize band definition
        
        Args:
            bands: Custom band list
            preset: Preset name if bands is None
        """
        if bands is not None:
            self.bands = bands
        else:
            self.bands = self.PRESETS.get(preset, self.DEFAULT_BANDS).copy()
        
        self._validate_bands()
    
    def _validate_bands(self):
        """Validate band definitions"""
        # Check for overlaps and gaps
        sorted_bands = sorted(self.bands, key=lambda b: b.low_freq)
        
        for i in range(len(sorted_bands) - 1):
            current = sorted_bands[i]
            next_band = sorted_bands[i + 1]
            
            if current.high_freq > next_band.low_freq:
                raise ValueError(f"Bands overlap: {current.name} and {next_band.name}")
            elif current.high_freq < next_band.low_freq:
                print(f"Warning: Gap between {current.name} and {next_band.name}")
    
    @classmethod
    def from_string(cls, band_string: str) -> 'BandDefinition':
        """
        Create BandDefinition from string format
        
        Args:
            band_string: Format "20-80,80-250,250-2000,2000-6000,6000-20000"
            
        Returns:
            BandDefinition instance
        """
        bands = []
        band_ranges = band_string.split(',')
        
        band_names = ["Band" + str(i+1) for i in range(len(band_ranges))]
        colors = ["#FF6B6B", "#FFA500", "#4ECDC4", "#45B7D1", "#96CEB4", 
                 "#DDA0DD", "#FFFF99", "#87CEEB"]
        
        for i, range_str in enumerate(band_ranges):
            low, high = map(float, range_str.split('-'))
            name = band_names[i] if i < len(band_names) else f"Band{i+1}"
            color = colors[i % len(colors)]
            bands.append(FrequencyBand(name, low, high, color))
        
        return cls(bands=bands)
    
    def get_fft_bins(self, n_fft: int, sample_rate: int) -> Dict[str, Tuple[int, int]]:
        """
        Calculate FFT bin indices for each band
        
        Args:
            n_fft: FFT size
            sample_rate: Sample rate
            
        Returns:
            Dictionary mapping band names to (start_bin, end_bin) tuples
        """
        freq_bins = np.fft.rfftfreq(n_fft, 1/sample_rate)
        bin_mapping = {}
        
        for band in self.bands:
            # Find bins within band frequency range
            start_bin = np.searchsorted(freq_bins, band.low_freq)
            end_bin = np.searchsorted(freq_bins, band.high_freq)
            
            # Ensure at least one bin per band
            if start_bin >= end_bin:
                end_bin = start_bin + 1
            
            bin_mapping[band.name] = (start_bin, end_bin)
        
        return bin_mapping
    
    def get_center_frequencies(self) -> Dict[str, float]:
        """
        Get geometric center frequency for each band
        
        Returns:
            Dictionary mapping band names to center frequencies
        """
        centers = {}
        for band in self.bands:
            # Geometric mean for logarithmic frequency scale
            centers[band.name] = np.sqrt(band.low_freq * band.high_freq)
        return centers
    
    def get_band_widths(self) -> Dict[str, float]:
        """
        Get bandwidth for each band
        
        Returns:
            Dictionary mapping band names to bandwidths in Hz
        """
        widths = {}
        for band in self.bands:
            widths[band.name] = band.high_freq - band.low_freq
        return widths
    
    def get_band_for_frequency(self, frequency: float) -> Optional[FrequencyBand]:
        """
        Find which band contains a given frequency
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            FrequencyBand containing the frequency, or None
        """
        for band in self.bands:
            if band.contains(frequency):
                return band
        return None
    
    def get_band_names(self) -> List[str]:
        """Get list of band names"""
        return [band.name for band in self.bands]
    
    def get_band_colors(self) -> List[str]:
        """Get list of band colors for visualization"""
        return [band.color for band in self.bands]
    
    def get_band_limits(self) -> Tuple[float, float]:
        """Get overall frequency range covered by bands"""
        min_freq = min(band.low_freq for band in self.bands)
        max_freq = max(band.high_freq for band in self.bands)
        return min_freq, max_freq
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'bands': [
                {
                    'name': band.name,
                    'low_freq': band.low_freq,
                    'high_freq': band.high_freq,
                    'color': band.color
                }
                for band in self.bands
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BandDefinition':
        """Create from dictionary"""
        bands = [
            FrequencyBand(
                name=b['name'],
                low_freq=b['low_freq'],
                high_freq=b['high_freq'],
                color=b.get('color', '#808080')
            )
            for b in data['bands']
        ]
        return cls(bands=bands)
    
    def adjust_band_limit(self, band_name: str, 
                          limit_type: str, 
                          new_freq: float):
        """
        Adjust a band's frequency limit
        
        Args:
            band_name: Name of band to adjust
            limit_type: 'low' or 'high'
            new_freq: New frequency limit
        """
        for i, band in enumerate(self.bands):
            if band.name == band_name:
                if limit_type == 'low':
                    # Check it doesn't overlap with previous band
                    if i > 0 and new_freq < self.bands[i-1].high_freq:
                        raise ValueError("Would overlap with previous band")
                    band.low_freq = new_freq
                elif limit_type == 'high':
                    # Check it doesn't overlap with next band
                    if i < len(self.bands)-1 and new_freq > self.bands[i+1].low_freq:
                        raise ValueError("Would overlap with next band")
                    band.high_freq = new_freq
                break
        
        self._validate_bands()


def calculate_octave_bands(start_freq: float = 31.5, 
                           num_bands: int = 10) -> List[FrequencyBand]:
    """
    Calculate octave or fractional octave bands
    
    Args:
        start_freq: Starting frequency
        num_bands: Number of bands
        
    Returns:
        List of FrequencyBand objects
    """
    bands = []
    freq = start_freq
    
    for i in range(num_bands):
        low_freq = freq
        high_freq = freq * 2  # Octave spacing
        
        if high_freq > 20000:
            high_freq = 20000
        
        bands.append(
            FrequencyBand(
                name=f"{int(freq)} Hz",
                low_freq=low_freq,
                high_freq=high_freq
            )
        )
        
        freq = high_freq
        if freq >= 20000:
            break
    
    return bands


def calculate_third_octave_bands() -> List[FrequencyBand]:
    """
    Calculate standard 1/3 octave bands from 20 Hz to 20 kHz
    
    Returns:
        List of FrequencyBand objects
    """
    # Standard 1/3 octave center frequencies
    center_freqs = [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
        2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000
    ]
    
    bands = []
    for fc in center_freqs:
        # 1/3 octave band edges
        low_freq = fc / (2 ** (1/6))
        high_freq = fc * (2 ** (1/6))
        
        bands.append(
            FrequencyBand(
                name=f"{fc} Hz",
                low_freq=low_freq,
                high_freq=high_freq
            )
        )
    
    return bands