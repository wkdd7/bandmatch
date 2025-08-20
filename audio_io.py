"""
Audio I/O module for BandMatch
Handles loading, resampling, and channel processing
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union
import subprocess
import tempfile
import os


class AudioLoader:
    """Audio file loading and preprocessing"""
    
    SUPPORTED_FORMATS = {'.wav', '.aiff', '.aif', '.flac', '.mp3', '.m4a', '.ogg'}
    
    def __init__(self, target_sr: int = 48000):
        """
        Initialize audio loader
        
        Args:
            target_sr: Target sample rate for standardization
        """
        self.target_sr = target_sr
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          check=True)
            self.ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False
            print("Warning: FFmpeg not found. MP3/M4A support limited.")
        return self.ffmpeg_available
    
    def load_audio(self, 
                   file_path: Union[str, Path],
                   mono: bool = True,
                   normalize: bool = False) -> Tuple[np.ndarray, int, dict]:
        """
        Load audio file and optionally convert to mono
        
        Args:
            file_path: Path to audio file
            mono: Convert to mono if True
            normalize: Normalize to [-1, 1] range
            
        Returns:
            Tuple of (audio_data, sample_rate, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        metadata = {
            'filename': file_path.name,
            'format': file_path.suffix.lower(),
            'original_sr': None,
            'duration': None,
            'channels': None
        }
        
        # Try soundfile first (for WAV/AIFF/FLAC)
        if file_path.suffix.lower() in {'.wav', '.aiff', '.aif', '.flac'}:
            try:
                audio, sr = sf.read(str(file_path))
                # soundfile returns (frames, channels) for stereo
                if audio.ndim == 2:
                    metadata['channels'] = audio.shape[1]
                else:
                    metadata['channels'] = 1
                    
            except Exception as e:
                print(f"Soundfile failed, trying librosa: {e}")
                audio, sr = librosa.load(str(file_path), sr=None, mono=False)
                if audio.ndim == 2:
                    metadata['channels'] = audio.shape[0]
                    audio = audio.T  # librosa returns (channels, frames)
                else:
                    metadata['channels'] = 1
        else:
            # Use librosa for MP3/M4A/OGG
            audio, sr = librosa.load(str(file_path), sr=None, mono=False)
            if audio.ndim == 2:
                metadata['channels'] = audio.shape[0]
                audio = audio.T  # librosa returns (channels, frames)
            else:
                metadata['channels'] = 1
        
        metadata['original_sr'] = sr
        metadata['duration'] = len(audio) / sr
        
        # Convert to mono if requested
        if mono and audio.ndim == 2:
            audio = self.to_mono(audio)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = self.resample(audio, sr, self.target_sr)
            sr = self.target_sr
        
        # Normalize if requested
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio, sr, metadata
    
    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono
        
        Args:
            audio: Audio data (frames, channels) or (frames,)
            
        Returns:
            Mono audio data
        """
        if audio.ndim == 1:
            return audio
        
        # Average channels
        return np.mean(audio, axis=1)
    
    def resample(self, 
                 audio: np.ndarray, 
                 orig_sr: int, 
                 target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        if audio.ndim == 1:
            # Mono
            return librosa.resample(audio, 
                                   orig_sr=orig_sr, 
                                   target_sr=target_sr)
        else:
            # Stereo - resample each channel
            resampled = []
            for ch in range(audio.shape[1]):
                resampled.append(
                    librosa.resample(audio[:, ch], 
                                   orig_sr=orig_sr, 
                                   target_sr=target_sr)
                )
            return np.stack(resampled, axis=1)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def save_audio(self, 
                   audio: np.ndarray, 
                   file_path: Union[str, Path], 
                   sample_rate: int = None):
        """
        Save audio to file
        
        Args:
            audio: Audio data
            file_path: Output file path
            sample_rate: Sample rate (uses target_sr if None)
        """
        if sample_rate is None:
            sample_rate = self.target_sr
        
        file_path = Path(file_path)
        sf.write(str(file_path), audio, sample_rate)
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get audio file information without loading full data
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        file_path = Path(file_path)
        info = sf.info(str(file_path))
        
        return {
            'filename': file_path.name,
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
            'format': info.format,
            'subtype': info.subtype
        }


def validate_audio_length(audio: np.ndarray, 
                         sr: int, 
                         min_duration: float = 10.0) -> Tuple[bool, str]:
    """
    Validate audio length
    
    Args:
        audio: Audio data
        sr: Sample rate
        min_duration: Minimum duration in seconds
        
    Returns:
        Tuple of (is_valid, message)
    """
    duration = len(audio) / sr
    
    if duration < min_duration:
        return False, f"Audio too short ({duration:.1f}s < {min_duration}s). Results may be unreliable."
    
    return True, f"Audio duration: {duration:.1f}s"


def batch_load_audio(file_paths: list, 
                    loader: Optional[AudioLoader] = None,
                    **kwargs) -> list:
    """
    Load multiple audio files
    
    Args:
        file_paths: List of audio file paths
        loader: AudioLoader instance (creates new if None)
        **kwargs: Additional arguments for load_audio
        
    Returns:
        List of (audio, sr, metadata) tuples
    """
    if loader is None:
        loader = AudioLoader()
    
    results = []
    for path in file_paths:
        try:
            result = loader.load_audio(path, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            results.append(None)
    
    return results