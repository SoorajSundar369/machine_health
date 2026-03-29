"""
Data loading and preprocessing module for machine sound analysis.
Handles audio file loading, standardization, and dataset creation.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict
import csv


class AudioDataLoader:
    """Load and preprocess audio data."""
    
    def __init__(self, sr: int = 22050, duration: float = 2.0):
        """
        Initialize audio data loader.
        
        Args:
            sr: Target sampling rate (Hz)
            duration: Target duration (seconds) - pad or truncate to this
        """
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load audio file and standardize.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio waveform of shape (self.samples,)
        """
        try:
            y, sr_loaded = librosa.load(filepath, sr=self.sr)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
        
        # Standardize duration
        y = self.standardize_duration(y)
        
        # Normalize amplitude
        y = self.normalize_amplitude(y)
        
        return y
    
    def standardize_duration(self, y: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to target duration.
        
        Args:
            y: Audio waveform
            
        Returns:
            Audio waveform of shape (self.samples,)
        """
        if len(y) < self.samples:
            # Pad with zeros
            padding = self.samples - len(y)
            y = np.pad(y, (0, padding), mode='constant', constant_values=0)
        elif len(y) > self.samples:
            # Truncate
            y = y[:self.samples]
        
        return y
    
    def normalize_amplitude(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1] range.
        
        Args:
            y: Audio waveform
            
        Returns:
            Normalized audio waveform
        """
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        
        return y
    
    def load_dataset_from_directory(
        self,
        data_dir: str,
        label: int = 1
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load all audio files from directory and assign labels.
        
        Args:
            data_dir: Directory containing audio files
            label: Label to assign to all files from this directory
            
        Returns:
            Tuple of (audio_waveforms, labels, filenames)
        """
        audio_files = []
        labels = []
        filenames = []
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        file_list = sorted([
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in audio_extensions
        ])
        
        for filename in file_list:
            filepath = os.path.join(data_dir, filename)
            y = self.load_audio(filepath)
            
            if y is not None:
                audio_files.append(y)
                labels.append(label)
                filenames.append(filename)
        
        return audio_files, labels, filenames
    
    def save_audio(self, y: np.ndarray, filepath: str) -> None:
        """
        Save audio waveform to file.
        
        Args:
            y: Audio waveform
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sf.write(filepath, y, self.sr)


class DatasetBuilder:
    """Build train/val/test datasets from audio files."""
    
    def __init__(self, sr: int = 22050, duration: float = 2.0, test_split: float = 0.2, val_split: float = 0.1):
        """
        Initialize dataset builder.
        
        Args:
            sr: Sampling rate
            duration: Audio duration
            test_split: Fraction of data for testing (0.0-1.0)
            val_split: Fraction of training data for validation (0.0-1.0)
        """
        self.loader = AudioDataLoader(sr=sr, duration=duration)
        self.sr = sr
        self.duration = duration
        self.test_split = test_split
        self.val_split = val_split
    
    def build_from_snr_dataset(
        self,
        data_root: str = "data/synthetic",
        snr_levels: List[int] = [20, 10, 5, 0]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build datasets from synthetic SNR-based directory structure.
        
        Directory structure expected:
            data_root/
            ├── clean/
            │   ├── normal/
            │   └── fault/
            └── noisy/
                ├── snr_20db/
                │   ├── normal/
                │   └── fault/
                └── snr_5db/
                    ├── normal/
                    └── fault/
        
        Args:
            data_root: Root directory of dataset
            snr_levels: SNR levels to include
            
        Returns:
            Dictionary with dataset splits and features
        """
        datasets = {}
        
        # Load clean data
        clean_normal_dir = os.path.join(data_root, "clean", "normal")
        clean_fault_dir = os.path.join(data_root, "clean", "fault")
        
        normal_sounds, normal_labels, normal_files = self.loader.load_dataset_from_directory(
            clean_normal_dir, label=0
        )
        fault_sounds, fault_labels, fault_files = self.loader.load_dataset_from_directory(
            clean_fault_dir, label=1
        )
        
        # Combine
        all_sounds = normal_sounds + fault_sounds
        all_labels = normal_labels + fault_labels
        all_files = normal_files + fault_files
        
        # Create splits
        datasets['clean'] = self._create_splits(all_sounds, all_labels, all_files)
        
        # Load noisy data at each SNR level
        for snr_db in snr_levels:
            snr_key = f'snr_{snr_db}db'
            noisy_normal_dir = os.path.join(data_root, "noisy", snr_key, "normal")
            noisy_fault_dir = os.path.join(data_root, "noisy", snr_key, "fault")
            
            if os.path.exists(noisy_normal_dir) and os.path.exists(noisy_fault_dir):
                normal_sounds, normal_labels, normal_files = self.loader.load_dataset_from_directory(
                    noisy_normal_dir, label=0
                )
                fault_sounds, fault_labels, fault_files = self.loader.load_dataset_from_directory(
                    noisy_fault_dir, label=1
                )
                
                all_sounds = normal_sounds + fault_sounds
                all_labels = normal_labels + fault_labels
                all_files = normal_files + fault_files
                
                datasets[snr_key] = self._create_splits(all_sounds, all_labels, all_files)
        
        return datasets
    
    def _create_splits(
        self,
        sounds: List[np.ndarray],
        labels: List[int],
        filenames: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create train/val/test splits from data.
        
        Args:
            sounds: List of audio waveforms
            labels: List of labels
            filenames: List of filenames
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing data arrays
        """
        # Shuffle and split
        indices = np.random.permutation(len(sounds))
        
        # Test split
        test_idx = int(len(indices) * self.test_split)
        train_val_indices = indices[test_idx:]
        test_indices = indices[:test_idx]
        
        # Validation split
        val_idx = int(len(train_val_indices) * self.val_split)
        val_indices = train_val_indices[:val_idx]
        train_indices = train_val_indices[val_idx:]
        
        # Create arrays
        def create_data_dict(idx_list):
            return {
                'audio': np.array([sounds[i] for i in idx_list]),
                'labels': np.array([labels[i] for i in idx_list]),
                'filenames': np.array([filenames[i] for i in idx_list])
            }
        
        return {
            'train': create_data_dict(train_indices),
            'val': create_data_dict(val_indices),
            'test': create_data_dict(test_indices)
        }
    
    def save_dataset_splits(
        self,
        datasets: Dict[str, Dict[str, np.ndarray]],
        output_dir: str = "data/processed"
    ) -> None:
        """
        Save dataset splits to disk as numpy files.
        
        Args:
            datasets: Dictionary of datasets
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for snr_key, splits in datasets.items():
            snr_dir = os.path.join(output_dir, snr_key)
            os.makedirs(snr_dir, exist_ok=True)
            
            for split_name, data in splits.items():
                split_dir = os.path.join(snr_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)
                
                np.save(os.path.join(split_dir, 'audio.npy'), data['audio'])
                np.save(os.path.join(split_dir, 'labels.npy'), data['labels'])
                np.save(os.path.join(split_dir, 'filenames.npy'), data['filenames'])
        
        print(f"✓ Dataset splits saved to {output_dir}")
    
    def load_dataset_splits(self, input_dir: str = "data/processed") -> Dict:
        """
        Load dataset splits from disk.
        
        Args:
            input_dir: Input directory containing saved splits
            
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        
        for snr_dir in os.listdir(input_dir):
            snr_path = os.path.join(input_dir, snr_dir)
            if not os.path.isdir(snr_path):
                continue
            
            datasets[snr_dir] = {}
            
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(snr_path, split)
                if os.path.exists(split_path):
                    datasets[snr_dir][split] = {
                        'audio': np.load(os.path.join(split_path, 'audio.npy')),
                        'labels': np.load(os.path.join(split_path, 'labels.npy')),
                        'filenames': np.load(os.path.join(split_path, 'filenames.npy'))
                    }
        
        return datasets


class AudioDataset:
    """PyTorch-style dataset for batch loading with feature extraction."""
    
    def __init__(
        self,
        audio_waveforms: np.ndarray,
        labels: np.ndarray,
        filenames: np.ndarray = None,
        feature_extractor=None
    ):
        """
        Initialize dataset.
        
        Args:
            audio_waveforms: Array of audio waveforms (n_samples, duration_samples)
            labels: Array of labels (n_samples,)
            filenames: Optional array of filenames
            feature_extractor: Optional feature extractor (for on-the-fly extraction)
        """
        self.audio = audio_waveforms
        self.labels = labels
        self.filenames = filenames
        self.feature_extractor = feature_extractor
    
    def __len__(self) -> int:
        return len(self.audio)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get single item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with 'audio', 'label', and optionally 'features' and 'filename'
        """
        item = {
            'audio': self.audio[idx],
            'label': self.labels[idx]
        }
        
        if self.filenames is not None:
            item['filename'] = self.filenames[idx]
        
        if self.feature_extractor is not None:
            item['features'] = self.feature_extractor.extract_mel_mfcc_combined(self.audio[idx])
        
        return item


# Convenience function
def load_and_split_dataset(
    data_root: str = "data/synthetic",
    snr_levels: List[int] = [20, 10, 5, 0],
    sr: int = 22050,
    duration: float = 2.0
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Load dataset, create splits, and return.
    
    Args:
        data_root: Root directory
        snr_levels: SNR levels to load
        sr: Sampling rate
        duration: Audio duration
        
    Returns:
        Dictionary of datasets with splits
    """
    builder = DatasetBuilder(sr=sr, duration=duration)
    datasets = builder.build_from_snr_dataset(data_root, snr_levels)
    
    print(f"✓ Loaded datasets:")
    for snr_key, splits in datasets.items():
        print(f"  {snr_key}:")
        for split_name, data in splits.items():
            print(f"    {split_name}: {len(data['audio'])} samples")
    
    return datasets
