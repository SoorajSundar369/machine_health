"""
Audio source separation module using DEMUCS for the cocktail party problem.
Separates machine sound from background noise in mixed audio.
"""

import numpy as np
import torch
import torchaudio
from typing import Tuple, Dict, Optional
import os


class DEMUCSSeparator:
    """
    Separate machine sound from background noise using DEMUCS model.
    
    DEMUCS (Hybrid Transformer Demucs) is pre-trained on music source separation
    and generalizes well to machinery sounds due to spectral patterns.
    """
    
    def __init__(self, model_name: str = 'htdemucs', device: str = 'cpu'):
        """
        Initialize DEMUCS separator.
        
        Args:
            model_name: DEMUCS model to use ('htdemucs', 'htdemucs_ft', 'hdemucs_mmi')
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.loaded = False
        
        try:
            self._load_model()
        except Exception as e:
            print(f"Warning: Could not load DEMUCS model: {e}")
            print("Make sure to install: pip install demucs")
    
    def _load_model(self):
        """Load pre-trained DEMUCS model."""
        try:
            from demucs.pretrained import get_model
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"✓ DEMUCS model '{self.model_name}' loaded successfully on {self.device}")
        except ImportError:
            raise ImportError(
                "DEMUCS not installed. Install with: pip install demucs"
            )
    
    def separate(
        self,
        waveform: np.ndarray,
        sr: int = 22050
    ) -> Dict[str, np.ndarray]:
        """
        Separate machine sound from background noise.
        
        Args:
            waveform: Audio waveform (numpy array) shape (n_samples,)
            sr: Sampling rate (Hz)
            
        Returns:
            Dictionary with keys:
            - 'machine': Isolated machine sound
            - 'noise': Isolated background noise
            - 'mixture': Original waveform
        """
        if not self.loaded:
            return self._fallback_separation(waveform)
        
        try:
            # Convert numpy to torch tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, n_samples)
            waveform_tensor = waveform_tensor.to(self.device)
            
            # Add channel dimension if needed
            if waveform_tensor.dim() == 2:
                waveform_tensor = waveform_tensor.unsqueeze(0)  # (1, 1, n_samples)
            
            with torch.no_grad():
                # DEMUCS outputs 4 stems: [drums, bass, vocals, other]
                # We treat 'other' as machine sound and combine rest as noise
                separated = self.model(waveform_tensor)  # (1, 4, n_samples)
            
            # Convert back to numpy
            separated_np = separated.squeeze(0).cpu().numpy()  # (4, n_samples)
            
            # Machine sound = 'other' stem (index 3)
            machine_sound = separated_np[3, :].astype(np.float32)
            
            # Noise = combine drums, bass, vocals (indices 0, 1, 2)
            noise = np.mean(separated_np[0:3, :], axis=0).astype(np.float32)
            
            return {
                'machine': machine_sound,
                'noise': noise,
                'mixture': waveform.astype(np.float32)
            }
        
        except Exception as e:
            print(f"Error during DEMUCS separation: {e}")
            return self._fallback_separation(waveform)
    
    def _fallback_separation(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fallback separation using spectral masking (Wiener filter).
        Used when DEMUCS is unavailable.
        """
        print("Using fallback spectral masking (Wiener filter) for separation...")
        
        # Estimate noise from quieter segments
        frame_length = 512
        n_frames = len(waveform) // frame_length
        
        frame_energies = []
        for i in range(n_frames):
            frame = waveform[i*frame_length:(i+1)*frame_length]
            energy = np.sqrt(np.mean(frame**2))
            frame_energies.append(energy)
        
        # Estimate noise power from quietest 10% of frames
        noise_energy = np.percentile(frame_energies, 10)
        signal_energy = np.sqrt(np.mean(waveform**2))
        
        # Simple attenuation of noise
        attenuation = max(0.1, 1.0 - (noise_energy / (signal_energy + 1e-8)))
        machine_sound = waveform * attenuation
        noise = waveform * (1.0 - attenuation)
        
        return {
            'machine': machine_sound.astype(np.float32),
            'noise': noise.astype(np.float32),
            'mixture': waveform.astype(np.float32)
        }


class SpectralMasking:
    """
    Simple spectral masking approach using Wiener filtering.
    Useful for quick prototyping before DEMUCS.
    """
    
    @staticmethod
    def wiener_filter(
        waveform: np.ndarray,
        noise_power: float = 0.01,
        frame_length: int = 512,
        hop_length: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Wiener filtering to separate signal from noise.
        
        Args:
            waveform: Input audio waveform
            noise_power: Estimated noise power
            frame_length: Window length for STFT
            hop_length: Frame hop length
            
        Returns:
            Tuple of (cleaned_signal, estimated_noise)
        """
        import librosa
        
        # Compute STFT
        stft = librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length)
        spec = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum (from quietest frames)
        noise_spec = np.percentile(spec, 20, axis=1, keepdims=True)
        
        # Wiener filter mask
        signal_power = spec ** 2
        total_power = signal_power + noise_power
        mask = signal_power / (total_power + 1e-8)
        
        # Apply mask to spectrogram
        cleaned_spec = mask * spec
        
        # Reconstruct with original phase
        cleaned_stft = cleaned_spec * np.exp(1j * phase)
        cleaned_signal = librosa.istft(cleaned_stft, hop_length=hop_length)
        
        # Estimate noise
        noise_spec_cleaned = (1.0 - mask) * spec
        noise_stft = noise_spec_cleaned * np.exp(1j * phase)
        estimated_noise = librosa.istft(noise_stft, hop_length=hop_length)
        
        return cleaned_signal, estimated_noise
    
    @staticmethod
    def spectral_subtraction(
        waveform: np.ndarray,
        noise_factor: float = 1.0,
        frame_length: int = 512,
        hop_length: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spectral subtraction to remove noise.
        
        Args:
            waveform: Input audio
            noise_factor: Scaling factor for noise subtraction (0-2)
            frame_length: Window length
            hop_length: Frame hop length
            
        Returns:
            Tuple of (cleaned_signal, estimated_noise)
        """
        import librosa
        
        # Compute STFT
        stft = librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length)
        spec = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum
        noise_spec = np.percentile(spec, 20, axis=1, keepdims=True)
        
        # Spectral subtraction
        cleaned_spec = np.maximum(spec - noise_factor * noise_spec, 0.1 * spec)
        
        # Reconstruct
        cleaned_stft = cleaned_spec * np.exp(1j * phase)
        cleaned_signal = librosa.istft(cleaned_stft, hop_length=hop_length)
        
        noise_stft = noise_spec * np.exp(1j * phase)
        estimated_noise = librosa.istft(noise_stft, hop_length=hop_length)
        
        return cleaned_signal, estimated_noise


def measure_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Measure Signal-to-Noise Ratio in dB.
    
    Args:
        signal: Signal component
        noise: Noise component
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_separation_quality(
    original_signal: np.ndarray,
    separated_signal: np.ndarray
) -> float:
    """
    Measure separation quality using Signal-to-Distortion Ratio (SDR).
    
    Args:
        original_signal: Clean original signal
        separated_signal: Separated/cleaned signal
        
    Returns:
        SDR in dB
    """
    # Align signals by removing initial transients
    if len(separated_signal) != len(original_signal):
        min_len = min(len(original_signal), len(separated_signal))
        original_signal = original_signal[:min_len]
        separated_signal = separated_signal[:min_len]
    
    # Calculate error
    error = original_signal - separated_signal
    
    signal_power = np.mean(original_signal ** 2)
    error_power = np.mean(error ** 2)
    
    if error_power < 1e-10:
        return float('inf')
    
    sdr = 10 * np.log10(signal_power / error_power)
    return sdr
