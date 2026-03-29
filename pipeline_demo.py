"""
Machine Health Audio Classification - Complete Pipeline
Demonstrates the full workflow from noisy audio to fault classification.
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.data import load_and_split_dataset, AudioDataLoader
from src.features import AudioFeatureExtractor
from src.separation import DEMUCSSeparator, SpectralMasking, measure_snr


def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def demonstrate_pipeline():
    """Run complete demonstration of the machine health system."""
    
    print_header("MACHINE HEALTH AUDIO CLASSIFICATION SYSTEM")
    print("\nThis demonstrates the complete pipeline for detecting machine faults")
    print("by solving the 'cocktail party problem' (background noise separation).\n")
    
    # PHASE 1: Load Data
    print_header("PHASE 1: DATA LOADING")
    
    print("\nLoading synthetic datasets at different SNR levels...")
    datasets = load_and_split_dataset(
        data_root='data/synthetic',
        snr_levels=[20, 10, 5, 0]
    )
    
    # Select a test sample
    test_snr = 10
    test_split = 'test'
    test_audio = datasets[f'snr_{test_snr}db'][test_split]['audio'][0]
    test_label = datasets[f'snr_{test_snr}db'][test_split]['labels'][0]
    label_name = 'NORMAL' if test_label == 0 else 'FAULT'
    
    print(f"\n✓ Test sample selected:")
    print(f"  Label: {label_name} machine sound")
    print(f"  SNR Level: {test_snr}dB (challenging noisy condition)")
    print(f"  Duration: {len(test_audio) / 22050:.2f} seconds")
    print(f"  Sample rate: 22050 Hz")
    
    # PHASE 2: Noise Separation (Cocktail Party Problem)
    print_header("PHASE 2: NOISE SEPARATION (COCKTAIL PARTY PROBLEM)")
    
    print("\nApplying separation techniques to isolate machine sound from noise...\n")
    
    # Method 1: Spectral Masking (always available)
    print("Method 1: Wiener Filter (Spectral Masking)")
    try:
        machine_wiener, noise_wiener = SpectralMasking.wiener_filter(test_audio)
        original_rms = np.sqrt(np.mean(test_audio**2))
        wiener_rms = np.sqrt(np.mean(machine_wiener**2))
        
        snr_original = measure_snr(test_audio, test_audio - machine_wiener)
        snr_wiener = measure_snr(machine_wiener, noise_wiener)
        
        print(f"  Status: ✓ Success")
        print(f"  Original SNR: ~{test_snr}dB")
        print(f"  Separated SNR: {snr_wiener:.2f}dB")
        print(f"  Improvement: {snr_wiener - test_snr:+.2f}dB")
        machine_separated = machine_wiener
        method_used = "Wiener Filter"
    except Exception as e:
        print(f"  Status: ✗ Failed ({e})")
        machine_separated = test_audio
        method_used = "None (using original)"
    
    # Method 2: DEMUCS (if available)
    print("\nMethod 2: DEMUCS (Pre-trained Transformer Model)")
    try:
        separator = DEMUCSSeparator(device='cpu')
        if separator.loaded:
            result = separator.separate(test_audio, sr=22050)
            machine_demucs = result['machine']
            noise_demucs = result['noise']
            
            snr_demucs = measure_snr(machine_demucs, noise_demucs)
            print(f"  Status: ✓ Success")
            print(f"  Separated SNR: {snr_demucs:.2f}dB")
            print(f"  Improvement: {snr_demucs - test_snr:+.2f}dB")
            
            if snr_demucs > snr_wiener:
                machine_separated = machine_demucs
                method_used = "DEMUCS"
                print(f"  → Using DEMUCS (better than Wiener: {snr_demucs:.2f} vs {snr_wiener:.2f})")
        else:
            print(f"  Status: ✗ Model not loaded")
            print(f"  Install with: pip install demucs")
    except Exception as e:
        print(f"  Status: ✗ Failed ({e})")
    
    print(f"\n📌 Selected method: {method_used}")
    
    # PHASE 3: Feature Extraction
    print_header("PHASE 3: FEATURE EXTRACTION")
    
    print("\nExtracting audio features from separated machine sound...\n")
    
    extractor = AudioFeatureExtractor(sr=22050, n_mels=128, n_mfcc=13)
    
    # Extract features
    mel_spec = extractor.extract_mel_spectrogram(machine_separated)
    mfcc = extractor.extract_mfcc(machine_separated)
    spectral_centroid = extractor.extract_spectral_centroid(machine_separated)
    spectral_rolloff = extractor.extract_spectral_rolloff(machine_separated)
    zcr = extractor.extract_zero_crossing_rate(machine_separated)
    rms = extractor.extract_rms_energy(machine_separated)
    
    print("✓ Features extracted:")
    print(f"  • Mel-spectrogram: {mel_spec.shape} (128 freq bins × time steps)")
    print(f"  • MFCC: {mfcc.shape} (13 coefficients × time steps)")
    print(f"  • Spectral centroid: {spectral_centroid.shape}")
    print(f"  • Spectral rolloff: {spectral_rolloff.shape}")
    print(f"  • Zero-crossing rate: {zcr.shape}")
    print(f"  • RMS energy: {rms.shape}")
    
    # Aggregate features
    combined_features = extractor.extract_mel_mfcc_combined(machine_separated)
    print(f"\n✓ Aggregated feature vector:")
    print(f"  • Combined (mel + mfcc): {combined_features.shape[0]} dimensions")
    print(f"  Ready for classification model")
    
    # PHASE 4: Visualization
    print_header("PHASE 4: VISUALIZATION")
    
    print("\nGenerating comparison visualizations...\n")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Machine Health Analysis: {label_name} Sound at SNR={test_snr}dB\nMethod: {method_used}',
                 fontsize=14, fontweight='bold')
    
    t = np.linspace(0, len(test_audio) / 22050, len(test_audio))
    time_slice = slice(0, int(0.5 * 22050))  # First 0.5 seconds
    
    # Row 0: Original noisy vs separated
    axes[0, 0].plot(t[time_slice], test_audio[time_slice], color='steelblue', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Original Noisy Signal (Waveform)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t[time_slice], machine_separated[time_slice], color='green', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Separated Machine Sound (Waveform)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 1: Spectrograms
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(test_audio)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=22050, ax=axes[1, 0], x_axis='time', y_axis='hz', cmap='Blues')
    axes[1, 0].set_title('Original Noisy - Spectrogram')
    
    D_separated = librosa.amplitude_to_db(np.abs(librosa.stft(machine_separated)), ref=np.max)
    librosa.display.specshow(D_separated, sr=22050, ax=axes[1, 1], x_axis='time', y_axis='hz', cmap='Greens')
    axes[1, 1].set_title('Separated Machine - Spectrogram')
    
    # Row 2: Feature representation
    librosa.display.specshow(mel_spec, sr=22050, hop_length=512, ax=axes[2, 0], x_axis='time', y_axis='mel', cmap='viridis')
    axes[2, 0].set_title('Mel-Spectrogram (Feature)')
    axes[2, 0].set_ylabel('Mel Frequency')
    
    librosa.display.specshow(mfcc, sr=22050, hop_length=512, ax=axes[2, 1], x_axis='time', cmap='plasma')
    axes[2, 1].set_title('MFCC Coefficients (Feature)')
    axes[2, 1].set_ylabel('Coefficient Index')
    
    plt.tight_layout()
    plt.savefig('pipeline_demonstration.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved: pipeline_demonstration.png")
    
    # Summary
    print_header("SUMMARY")
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           MACHINE HEALTH CLASSIFICATION PIPELINE           ║
╠═══════════════════════════════════════════════════════════╣
║                                                            ║
║  1. Input: Noisy machine sound (SNR={test_snr}dB)           ║
║  2. Separation Method: {method_used:<30} ║
║  3. Output: Isolated machine sound                        ║
║  4. Features Extracted: {combined_features.shape[0]} dimensional vector         ║
║  5. Ground Truth: {label_name} machine                            ║
║                                                            ║
║  Status: ✓ READY FOR CLASSIFICATION                       ║
║                                                            ║
║  Next: Train CNN classifier on separated audio features   ║
║        to distinguish NORMAL vs FAULT conditions          ║
║                                                            ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n✓ Full pipeline demonstration complete!")
    print("\nFiles generated:")
    print("  • pipeline_demonstration.png - Visualization")
    print("\nNext phase: PHASE 3 - CNN Classifier Training")


if __name__ == '__main__':
    try:
        demonstrate_pipeline()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
