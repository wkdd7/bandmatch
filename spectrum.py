"""
Spectrum analysis module for BandMatch
Implements STFT and band energy calculations
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
from bands import BandDefinition


class SpectrumAnalyzer:
    """Performs spectral analysis on audio signals"""
    
    def __init__(self, 
                 n_fft: int = 4096,
                 hop_length: Optional[int] = None,
                 window: str = 'hann',
                 band_definition: Optional[BandDefinition] = None):
        """
        Initialize spectrum analyzer
        
        Args:
            n_fft: FFT size
            hop_length: Hop length for STFT (default: n_fft // 4)
            window: Window function name
            band_definition: Band definition to use
        """
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.band_definition = band_definition or BandDefinition()
        
        # Pre-compute window
        self._window = signal.get_window(window, n_fft)
    
    def compute_stft(self, 
                     audio: np.ndarray, 
                     sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform
        
        Args:
            audio: Audio signal (mono)
            sample_rate: Sample rate
            
        Returns:
            Tuple of (frequencies, times, stft_matrix)
        """
        if audio.ndim > 1:
            raise ValueError("Audio must be mono for STFT")
        
        # Compute STFT using scipy
        frequencies, times, stft_matrix = signal.stft(
            audio,
            fs=sample_rate,
            window=self._window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            return_onesided=True
        )
        
        return frequencies, times, stft_matrix
    
    def compute_power_spectrum(self, 
                               stft_matrix: np.ndarray) -> np.ndarray:
        """
        Convert STFT to power spectrum
        
        Args:
            stft_matrix: Complex STFT matrix
            
        Returns:
            Power spectrum (magnitude squared)
        """
        return np.abs(stft_matrix) ** 2
    
    def compute_band_energy(self,
                           power_spectrum: np.ndarray,
                           sample_rate: int,
                           epsilon: float = 1e-10) -> Dict[str, np.ndarray]:
        """
        Calculate energy for each frequency band
        
        Args:
            power_spectrum: Power spectrum matrix (freq_bins x time_frames)
            sample_rate: Sample rate
            epsilon: Small value for numerical stability
            
        Returns:
            Dictionary mapping band names to energy arrays (per time frame)
        """
        # Get bin mapping for bands
        bin_mapping = self.band_definition.get_fft_bins(self.n_fft, sample_rate)
        
        band_energy = {}
        for band_name, (start_bin, end_bin) in bin_mapping.items():
            # Sum energy in band for each time frame
            energy = np.sum(power_spectrum[start_bin:end_bin, :], axis=0)
            # Add epsilon for numerical stability before log
            band_energy[band_name] = energy + epsilon
        
        return band_energy
    
    def band_energy_to_db(self, 
                         band_energy: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert band energy to dB scale
        
        Args:
            band_energy: Dictionary of band energies
            
        Returns:
            Dictionary of band energies in dB
        """
        band_db = {}
        for band_name, energy in band_energy.items():
            band_db[band_name] = 10 * np.log10(energy)
        return band_db
    
    def aggregate_time(self,
                       band_db: Dict[str, np.ndarray],
                       method: str = 'median') -> Dict[str, float]:
        """
        Aggregate band energy over time
        
        Args:
            band_db: Band energies in dB over time
            method: Aggregation method ('median', 'mean', 'percentile_95')
            
        Returns:
            Dictionary mapping band names to single dB values
        """
        aggregated = {}
        
        for band_name, db_values in band_db.items():
            if method == 'median':
                aggregated[band_name] = np.median(db_values)
            elif method == 'mean':
                aggregated[band_name] = np.mean(db_values)
            elif method == 'percentile_95':
                aggregated[band_name] = np.percentile(db_values, 95)
            elif method == 'percentile_75':
                aggregated[band_name] = np.percentile(db_values, 75)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated
    
    def analyze_audio(self,
                     audio: np.ndarray,
                     sample_rate: int,
                     aggregate_method: str = 'median') -> Dict[str, float]:
        """
        Complete spectral analysis pipeline
        
        Args:
            audio: Audio signal (mono)
            sample_rate: Sample rate
            aggregate_method: Time aggregation method
            
        Returns:
            Dictionary mapping band names to dB values
        """
        # Compute STFT
        frequencies, times, stft_matrix = self.compute_stft(audio, sample_rate)
        
        # Convert to power spectrum
        power_spectrum = self.compute_power_spectrum(stft_matrix)
        
        # Calculate band energy
        band_energy = self.compute_band_energy(power_spectrum, sample_rate)
        
        # Convert to dB
        band_db = self.band_energy_to_db(band_energy)
        
        # Aggregate over time
        aggregated_db = self.aggregate_time(band_db, aggregate_method)
        
        return aggregated_db
    
    def compute_spectral_centroid(self,
                                  power_spectrum: np.ndarray,
                                  frequencies: np.ndarray) -> np.ndarray:
        """
        Compute spectral centroid for each time frame
        
        Args:
            power_spectrum: Power spectrum matrix
            frequencies: Frequency array
            
        Returns:
            Spectral centroid for each time frame
        """
        # Weighted average of frequencies
        numerator = np.sum(frequencies[:, np.newaxis] * power_spectrum, axis=0)
        denominator = np.sum(power_spectrum, axis=0)
        
        # Avoid division by zero
        centroid = np.where(denominator > 0, numerator / denominator, 0)
        
        return centroid
    
    def compute_spectral_rolloff(self,
                                 power_spectrum: np.ndarray,
                                 frequencies: np.ndarray,
                                 rolloff_percent: float = 0.85) -> np.ndarray:
        """
        Compute spectral rolloff frequency
        
        Args:
            power_spectrum: Power spectrum matrix
            frequencies: Frequency array
            rolloff_percent: Percentage of total energy
            
        Returns:
            Rolloff frequency for each time frame
        """
        # Calculate cumulative energy
        cumsum = np.cumsum(power_spectrum, axis=0)
        total_energy = cumsum[-1, :]
        
        # Find rolloff point
        rolloff_bins = np.argmax(cumsum >= rolloff_percent * total_energy, axis=0)
        rolloff_freqs = frequencies[rolloff_bins]
        
        return rolloff_freqs
    
    def compute_spectral_features(self,
                                  audio: np.ndarray,
                                  sample_rate: int) -> Dict[str, float]:
        """
        Compute various spectral features
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        # Compute STFT
        frequencies, times, stft_matrix = self.compute_stft(audio, sample_rate)
        power_spectrum = self.compute_power_spectrum(stft_matrix)
        
        # Compute features
        centroid = self.compute_spectral_centroid(power_spectrum, frequencies)
        rolloff = self.compute_spectral_rolloff(power_spectrum, frequencies)
        
        # Aggregate
        features = {
            'spectral_centroid_mean': np.mean(centroid),
            'spectral_centroid_std': np.std(centroid),
            'spectral_rolloff_mean': np.mean(rolloff),
            'spectral_rolloff_std': np.std(rolloff)
        }
        
        return features


class MultiChannelAnalyzer:
    """Analyzer for stereo/multi-channel audio"""
    
    def __init__(self, analyzer: Optional[SpectrumAnalyzer] = None):
        """
        Initialize multi-channel analyzer
        
        Args:
            analyzer: SpectrumAnalyzer instance to use
        """
        self.analyzer = analyzer or SpectrumAnalyzer()
    
    def analyze_stereo(self,
                      audio: np.ndarray,
                      sample_rate: int,
                      aggregate_method: str = 'median') -> Dict[str, Dict[str, float]]:
        """
        Analyze stereo audio
        
        Args:
            audio: Stereo audio (samples x 2)
            sample_rate: Sample rate
            aggregate_method: Time aggregation method
            
        Returns:
            Dictionary with 'left', 'right', 'mono', 'diff' analyses
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Audio must be stereo (samples x 2)")
        
        # Split channels
        left = audio[:, 0]
        right = audio[:, 1]
        mono = (left + right) / 2
        diff = left - right  # Side signal
        
        # Analyze each
        results = {
            'left': self.analyzer.analyze_audio(left, sample_rate, aggregate_method),
            'right': self.analyzer.analyze_audio(right, sample_rate, aggregate_method),
            'mono': self.analyzer.analyze_audio(mono, sample_rate, aggregate_method),
            'diff': self.analyzer.analyze_audio(diff, sample_rate, aggregate_method)
        }
        
        return results
    
    def compute_stereo_width(self, 
                            stereo_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute stereo width for each band
        
        Args:
            stereo_results: Results from analyze_stereo
            
        Returns:
            Dictionary mapping band names to width values
        """
        width = {}
        
        mono_db = stereo_results['mono']
        diff_db = stereo_results['diff']
        
        for band_name in mono_db.keys():
            # Stereo width as ratio of side to mid
            width[band_name] = diff_db[band_name] - mono_db[band_name]
        
        return width


def validate_spectrum_parameters(n_fft: int, 
                                sample_rate: int,
                                audio_length: int) -> Tuple[bool, str]:
    """
    Validate spectrum analysis parameters
    
    Args:
        n_fft: FFT size
        sample_rate: Sample rate
        audio_length: Audio length in samples
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check FFT size
    if n_fft < 512:
        return False, f"FFT size too small: {n_fft} (minimum 512)"
    
    if n_fft > audio_length:
        return False, f"FFT size larger than audio: {n_fft} > {audio_length}"
    
    # Check frequency resolution
    freq_resolution = sample_rate / n_fft
    if freq_resolution > 50:
        return False, f"Poor frequency resolution: {freq_resolution:.1f} Hz"
    
    return True, f"Valid parameters (resolution: {freq_resolution:.1f} Hz)"