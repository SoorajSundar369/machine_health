import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import csv

# Create legacy directories for backward compatibility
os.makedirs("data/normal", exist_ok=True)
os.makedirs("data/fault", exist_ok=True)

# Create new structured directories
os.makedirs("data/synthetic/clean/normal", exist_ok=True)
os.makedirs("data/synthetic/clean/fault", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_20db/normal", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_20db/fault", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_10db/normal", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_10db/fault", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_5db/normal", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_5db/fault", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_0db/normal", exist_ok=True)
os.makedirs("data/synthetic/noisy/snr_0db/fault", exist_ok=True)

SR = 22050
DURATION = 2
NUM_SAMPLES = 20

def generate_normal(filename=None):
    """Generate clean normal machine sound"""
    t = np.linspace(0, DURATION, SR * DURATION)
    signal = (
        0.5 * np.sin(2 * np.pi * 50 * t) +
        0.3 * np.sin(2 * np.pi * 100 * t) +
        0.05 * np.random.randn(len(t))
    )
    if filename:
        sf.write(filename, signal, SR)
    return signal

def generate_fault(filename=None):
    """Generate faulty machine sound with abnormal frequency"""
    t = np.linspace(0, DURATION, SR * DURATION)
    signal = (
        0.5 * np.sin(2 * np.pi * 50 * t) +
        0.3 * np.sin(2 * np.pi * 100 * t) +
        0.3 * np.sin(2 * np.pi * 374 * t) +  # abnormal frequency
        0.2 * np.random.randn(len(t))        # more noise = wear/fault
    )
    if filename:
        sf.write(filename, signal, SR)
    return signal

def generate_background_noise():
    """Generate background noise (simulates environmental/other machinery noise)"""
    # Simulate mixture of multiple noise sources
    t = np.linspace(0, DURATION, SR * DURATION)
    noise = (
        0.15 * np.sin(2 * np.pi * 150 * t) +      # other machine tone
        0.1 * np.sin(2 * np.pi * 220 * t) +       # background hum
        0.2 * np.random.randn(len(t)) +           # white noise (crowd, environment)
        0.05 * np.sin(2 * np.pi * np.random.uniform(200, 400, len(t)) * t)  # varying frequencies
    )
    return noise

def add_noise_at_snr(signal, snr_db):
    """
    Mix signal with background noise at target SNR.
    SNR = 10 * log10(P_signal / P_noise)
    """
    noise = generate_background_noise()
    
    # Normalize noise to match target SNR
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    
    # Calculate noise scaling factor
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(p_signal / (snr_linear * p_noise))
    noise_scaled = noise * noise_scale
    
    # Mix signal and noise
    noisy_signal = signal + noise_scaled
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 1.0:
        noisy_signal = noisy_signal / max_val
    
    return noisy_signal

def calculate_snr(signal, noise):
    """Calculate actual SNR between signal and noise"""
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0:
        return np.inf
    return 10 * np.log10(p_signal / p_noise)

print("=" * 60)
print("PHASE 1: Generating Synthetic Machine Sounds")
print("=" * 60)

# Store all sounds for visualization
normal_sounds = []
fault_sounds = []
metadata = []

print(f"\n1. Generating {NUM_SAMPLES} clean normal sounds...")
for i in range(NUM_SAMPLES):
    s = generate_normal(f"data/synthetic/clean/normal/normal_{i:03d}.wav")
    normal_sounds.append(s)
    metadata.append({"filename": f"normal_{i:03d}.wav", "label": "normal", "snr_db": "clean"})

print(f"2. Generating {NUM_SAMPLES} clean fault sounds...")
for i in range(NUM_SAMPLES):
    s = generate_fault(f"data/synthetic/clean/fault/fault_{i:03d}.wav")
    fault_sounds.append(s)
    metadata.append({"filename": f"fault_{i:03d}.wav", "label": "fault", "snr_db": "clean"})

# Legacy paths for backward compatibility
print(f"3. Creating legacy data directories...")
for i in range(NUM_SAMPLES):
    sf.write(f"data/normal/normal_{i}.wav", normal_sounds[i], SR)
    sf.write(f"data/fault/fault_{i}.wav", fault_sounds[i], SR)

# Generate noisy versions at different SNR levels
snr_levels = [20, 10, 5, 0]
print(f"\n4. Generating noisy versions at SNR levels: {snr_levels} dB...")
for snr_db in snr_levels:
    print(f"   SNR = {snr_db} dB...")
    for i, normal_signal in enumerate(normal_sounds):
        noisy = add_noise_at_snr(normal_signal, snr_db)
        sf.write(f"data/synthetic/noisy/snr_{snr_db}db/normal/normal_{i:03d}.wav", noisy, SR)
        metadata.append({"filename": f"normal_{i:03d}.wav", "label": "normal", "snr_db": snr_db})
    
    for i, fault_signal in enumerate(fault_sounds):
        noisy = add_noise_at_snr(fault_signal, snr_db)
        sf.write(f"data/synthetic/noisy/snr_{snr_db}db/fault/fault_{i:03d}.wav", noisy, SR)
        metadata.append({"filename": f"fault_{i:03d}.wav", "label": "fault", "snr_db": snr_db})

# Save metadata CSV
csv_path = "data/synthetic/metadata.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label", "snr_db"])
    writer.writeheader()
    writer.writerows(metadata)

print(f"\n✓ Dataset generation complete!")
print(f"  - {NUM_SAMPLES * 2} clean sounds (normal + fault)")
print(f"  - {NUM_SAMPLES * 2 * len(snr_levels)} noisy sounds ({len(snr_levels)} SNR levels)")
print(f"  - Metadata saved to {csv_path}")
print("=" * 60)

print("\n" + "=" * 60)
print("VISUALIZATION: Clean vs Noisy Sounds")
print("=" * 60)

t = np.linspace(0, DURATION, SR * DURATION)
time_slice = slice(0, int(0.5 * SR))  # Show first 0.5 seconds

# Plot 1: Waveforms - Normal (clean vs noisy at different SNR)
fig, axes = plt.subplots(5, 2, figsize=(14, 14))
fig.suptitle("Normal Machine Sound: Clean vs Noisy (Different SNR Levels)", fontsize=14, fontweight='bold')

# Clean normal
axes[0, 0].plot(t[time_slice], normal_sounds[0][time_slice], color='steelblue', linewidth=1.5)
axes[0, 0].set_title("Clean Normal — Waveform")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True, alpha=0.3)

# Spectrogram of clean normal
D_clean_normal = librosa.amplitude_to_db(np.abs(librosa.stft(normal_sounds[0])), ref=np.max)
librosa.display.specshow(D_clean_normal, sr=SR, ax=axes[0, 1], x_axis='time', y_axis='hz', cmap='Blues')
axes[0, 1].set_title("Clean Normal — Spectrogram")

# Noisy versions at different SNR
snr_display = [20, 10, 5, 0]
colors = ['green', 'orange', 'gold', 'red']
for idx, snr_db in enumerate(snr_display):
    noisy_normal = add_noise_at_snr(normal_sounds[0], snr_db)
    
    # Waveform
    axes[idx + 1, 0].plot(t[time_slice], noisy_normal[time_slice], color=colors[idx], linewidth=1, alpha=0.8)
    axes[idx + 1, 0].set_title(f"Noisy Normal (SNR={snr_db}dB) — Waveform")
    axes[idx + 1, 0].set_ylabel("Amplitude")
    axes[idx + 1, 0].grid(True, alpha=0.3)
    
    # Spectrogram
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_normal)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=SR, ax=axes[idx + 1, 1], x_axis='time', y_axis='hz', cmap='Blues')
    axes[idx + 1, 1].set_title(f"Noisy Normal (SNR={snr_db}dB) — Spectrogram")

plt.tight_layout()
plt.savefig("visualization_normal_vs_noisy.png", dpi=150, bbox_inches='tight')
print("✓ Chart saved: visualization_normal_vs_noisy.png")

# Plot 2: Fault vs Normal comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Normal vs Faulty: Clean Sounds Comparison", fontsize=14, fontweight='bold')

# Normal clean
axes[0, 0].plot(t[time_slice], normal_sounds[0][time_slice], color='steelblue', linewidth=1.5)
axes[0, 0].set_title("Clean Normal Machine Sound")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")

# Fault clean
axes[0, 1].plot(t[time_slice], fault_sounds[0][time_slice], color='tomato', linewidth=1.5)
axes[0, 1].set_title("Clean Faulty Machine Sound (Abnormal Frequency)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Amplitude")

# Spectrograms
D_normal = librosa.amplitude_to_db(np.abs(librosa.stft(normal_sounds[0])), ref=np.max)
D_fault = librosa.amplitude_to_db(np.abs(librosa.stft(fault_sounds[0])), ref=np.max)

im0 = librosa.display.specshow(D_normal, sr=SR, ax=axes[1, 0], x_axis='time', y_axis='hz', cmap='Blues')
axes[1, 0].set_title("Clean Normal — Spectrogram")

im1 = librosa.display.specshow(D_fault, sr=SR, ax=axes[1, 1], x_axis='time', y_axis='hz', cmap='Reds')
axes[1, 1].set_title("Clean Fault — Spectrogram (Extra peak at 374 Hz)")

plt.tight_layout()
plt.savefig("visualization_normal_vs_fault.png", dpi=150, bbox_inches='tight')
print("✓ Chart saved: visualization_normal_vs_fault.png")

plt.show()
print("=" * 60)
print("\n✓ PHASE 1 COMPLETE: All visualizations generated!")