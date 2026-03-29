"""
Feature extraction module for machine sound analysis.
Extracts MFCC, mel-spectrograms, spectral features, and others.
"""

import numpy as np
import librosa
from typing import Tuple, Dict, List


class AudioFeatureExtractor:
    """Extract various audio features for machine health classification."""
    
    def __init__(self, sr: int = 22050, n_mels: int = 128, n_mfcc: int = 13):
        """
        Initialize feature extractor.
        
        Args:
            sr: Sampling rate (Hz)
            n_mels: Number of mel-frequency bins
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram.
        
        Args:
            y: Audio waveform (numpy array)
            
        Returns:
            Mel-spectrogram of shape (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients).
        
        Args:
            y: Audio waveform
            
        Returns:
            MFCC of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc
        )
        return mfcc
    
    def extract_spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid (center of mass of the spectrum).
        
        Args:
            y: Audio waveform
            
        Returns:
            Spectral centroid of shape (time_steps,)
        """
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        return spec_centroid[0]  # Return 1D array
    
    def extract_spectral_rolloff(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral rolloff (frequency below which 85% of energy is concentrated).
        
        Args:
            y: Audio waveform
            
        Returns:
            Spectral rolloff of shape (time_steps,)
        """
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        return spec_rolloff[0]
    
    def extract_zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Extract zero crossing rate (measure of noisiness).
        
        Args:
            y: Audio waveform
            
        Returns:
            ZCR of shape (time_steps,)
        """
        zcr = librosa.feature.zero_crossing_rate(y)
        return zcr[0]
    
    def extract_rms_energy(self, y: np.ndarray) -> np.ndarray:
        """
        Extract RMS (Root Mean Square) energy.
        
        Args:
            y: Audio waveform
            
        Returns:
            RMS energy of shape (time_steps,)
        """
        rms = librosa.feature.rms(y=y)
        return rms[0]
    
    def extract_chroma_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract chroma features (identifies periodic components).
        
        Args:
            y: Audio waveform
            
        Returns:
            Chroma STFT of shape (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        return chroma
    
    def aggregate_features(self, feature_array: np.ndarray) -> np.ndarray:
        """
        Aggregate time-series features to fixed-size vector.
        Computes mean, std, min, max across time dimension.
        
        Args:
            feature_array: Feature array of shape (feature_dim, time_steps)
            
        Returns:
            Aggregated features of shape (feature_dim * 4,)
        """
        mean_feat = np.mean(feature_array, axis=1)
        std_feat = np.std(feature_array, axis=1)
        min_feat = np.min(feature_array, axis=1)
        max_feat = np.max(feature_array, axis=1)
        
        return np.concatenate([mean_feat, std_feat, min_feat, max_feat])
    
    def extract_all_features(self, y: np.ndarray, aggregate: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract all available features.
        
        Args:
            y: Audio waveform
            aggregate: If True, aggregate time-series features to fixed vectors
            
        Returns:
            Dictionary with feature names as keys and feature arrays as values
        """
        features = {}
        
        # Frequency-domain features
        features['mel_spectrogram'] = self.extract_mel_spectrogram(y)
        features['mfcc'] = self.extract_mfcc(y)
        features['chroma'] = self.extract_chroma_features(y)
        
        # Spectral features
        features['spectral_centroid'] = self.extract_spectral_centroid(y)
        features['spectral_rolloff'] = self.extract_spectral_rolloff(y)
        features['zero_crossing_rate'] = self.extract_zero_crossing_rate(y)
        features['rms_energy'] = self.extract_rms_energy(y)
        
        if aggregate:
            # Aggregate time-series features
            agg_features = {}
            for key, feat in features.items():
                if feat.ndim > 1:  # Has time dimension
                    agg_features[f'{key}_agg'] = self.aggregate_features(feat)
                else:
                    # 1D features - aggregate directly
                    agg_features[f'{key}_agg'] = np.array([
                        np.mean(feat), np.std(feat), np.min(feat), np.max(feat)
                    ])
            
            # Concatenate all aggregated features into single vector
            agg_vector = np.concatenate([v for v in agg_features.values()])
            agg_features['concatenated'] = agg_vector
            
            return agg_features
        
        return features
    
    def extract_mel_mfcc_combined(self, y: np.ndarray) -> np.ndarray:
        """
        Extract combined mel-spectrogram and MFCC features (recommended for CNN).
        Aggregates both for fixed-size output.
        
        Args:
            y: Audio waveform
            
        Returns:
            Combined feature vector (aggregated mel-spec + MFCC)
        """
        mel_spec = self.extract_mel_spectrogram(y)
        mfcc = self.extract_mfcc(y)
        
        mel_spec_agg = self.aggregate_features(mel_spec)
        mfcc_agg = self.aggregate_features(mfcc)
        
        return np.concatenate([mel_spec_agg, mfcc_agg])


def extract_features_batch(
    audio_files: List[str],
    sr: int = 22050,
    feature_type: str = 'mel_mfcc'
) -> np.ndarray:
    """
    Extract features from a batch of audio files.
    
    Args:
        audio_files: List of file paths
        sr: Sampling rate
        feature_type: Type of features ('mel_mfcc', 'mfcc', 'mel_spec')
        
    Returns:
        Feature matrix of shape (n_files, feature_dim)
    """
    extractor = AudioFeatureExtractor(sr=sr)
    features_list = []
    
    for filepath in audio_files:
        try:
            y, _ = librosa.load(filepath, sr=sr)
            
            if feature_type == 'mel_mfcc':
                features = extractor.extract_mel_mfcc_combined(y)
            elif feature_type == 'mfcc':
                mfcc = extractor.extract_mfcc(y)
                features = extractor.aggregate_features(mfcc)
            elif feature_type == 'mel_spec':
                mel_spec = extractor.extract_mel_spectrogram(y)
                features = extractor.aggregate_features(mel_spec)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features_list.append(features)
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    return np.array(features_list)
