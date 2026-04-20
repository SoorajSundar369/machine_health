"""
=============================================================
  MACHINE HEALTH MONITOR  —  Noise Filter Module
=============================================================
This module removes background noise from audio signals.

Technique used: SPECTRAL SUBTRACTION
  1. Look at the first 0.5 seconds of the recording (assumed quiet / pre-machine)
  2. Measure what frequencies are present — that's the background noise profile
  3. Subtract that noise profile from the entire recording
  4. Return the cleaned audio

This file is used by both train.py and predict.py.
You don't need to run this file directly.
"""

import numpy as np
import librosa


def clean_signal(audio, sr):
    """
    Removes stationary background noise from an audio signal.

    Parameters
    ----------
    audio : numpy array
        The raw audio signal (1D array of float values).
    sr : int
        Sample rate of the audio (e.g. 22050 means 22050 samples per second).

    Returns
    -------
    numpy array
        The cleaned audio signal, same length as input, as float32.

    How it works (plain English):
    ─────────────────────────────
    Imagine the audio is a mix of the machine sound (signal) and background
    noise (fans, traffic, other machines). The first 0.5 seconds are likely
    just noise before the machine kicks in. We measure that noise, then
    subtract it from the whole recording — like erasing the background.

    Steps:
    1. Convert both the noise sample and full audio to the frequency domain
       using STFT (Short-Time Fourier Transform). This shows how strong each
       frequency is at each moment in time.
    2. Average the noise frequencies across the 0.5s sample → noise profile.
    3. Subtract 2× the noise profile from the full signal (over-subtract a bit
       to be aggressive about cleaning).
    4. Clip to a minimum of 1% of original magnitude so we don't create silence.
    5. Convert back to time-domain audio.
    """

    # ── Guard: audio must be long enough for a noise sample ──────────────
    noise_length = int(0.5 * sr)   # 0.5 seconds worth of samples
    if len(audio) < noise_length:
        # Too short to estimate noise — return audio unchanged
        return audio.astype(np.float32)

    # ── Step 1: Isolate the noise sample (first 0.5 seconds) ─────────────
    noise_segment = audio[:noise_length]

    # ── Step 2: Convert to frequency domain with STFT ────────────────────
    # n_fft=1024 : chunk size (balances frequency resolution vs time resolution)
    # hop_length=256 : how much we slide the window each step
    stft_full  = librosa.stft(audio,          n_fft=1024, hop_length=256)
    stft_noise = librosa.stft(noise_segment,  n_fft=1024, hop_length=256)

    # Separate magnitude (strength) and phase (timing) of each frequency
    magnitude  = np.abs(stft_full)    # how strong each frequency is
    phase      = np.angle(stft_full)  # the timing component (needed to rebuild audio)

    # ── Step 3: Estimate noise profile ───────────────────────────────────
    # Average across all time frames → one noise level per frequency bin
    # keepdims=True keeps the shape so broadcasting works in subtraction
    noise_profile = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

    # ── Step 4: Subtract noise from signal ───────────────────────────────
    # Over-subtract by 2× (aggressive cleaning)
    # np.maximum(..., 0.01 * magnitude) floors at 1% to avoid silent gaps
    clean_magnitude = np.maximum(
        magnitude - 2.0 * noise_profile,
        0.01 * magnitude
    )

    # ── Step 5: Reconstruct the time-domain audio ─────────────────────────
    # Combine cleaned magnitude with original phase, then inverse STFT
    clean_stft = clean_magnitude * np.exp(1j * phase)
    cleaned    = librosa.istft(clean_stft, hop_length=256, length=len(audio))

    # ── Normalize volume ──────────────────────────────────────────────────
    peak = np.max(np.abs(cleaned))
    if peak > 0:
        cleaned = cleaned / peak * 0.9

    return cleaned.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY DETECTION MODULE - Spectral Signature Validation
# ═══════════════════════════════════════════════════════════════════════════════
"""
This module checks if known machine frequencies are present in the audio.
Used to distinguish between:
  - "No Sound" (silence)
  - "No Machine Sound" (random noise/speech, but no machine signature)
  - Known machine types (if characteristic frequencies are present)

Frequency Signatures (from training data analysis):
  - NORMAL:      1252 Hz (±300 Hz band: 952-1552 Hz)
  - ELECTRICAL:  3406 Hz (±300 Hz band: 3106-3706 Hz)
  - MECHANICAL:  2979 Hz (±300 Hz band: 2679-3279 Hz)
"""

from scipy import signal as scipy_signal


# Define characteristic frequency bands for each machine type
FREQUENCY_BANDS = {
    'normal':      {'center': 1252, 'margin': 300},      # 952-1552 Hz
    'electrical':  {'center': 3406, 'margin': 300},      # 3106-3706 Hz
    'mechanical':  {'center': 2979, 'margin': 300},      # 2679-3279 Hz
}


def get_frequency_spectrum(audio, sr, n_fft=2048):
    """
    Compute the power spectral density of audio.
    
    Parameters
    ----------
    audio : numpy array
        Audio signal
    sr : int
        Sample rate
    n_fft : int
        FFT size (default 2048)
    
    Returns
    -------
    tuple : (frequencies, power_spectrum)
        frequencies: frequency bins (Hz)
        power_spectrum: power at each frequency
    """
    # Use Welch's method for smoother spectrum estimate
    frequencies, pxx = scipy_signal.welch(audio, sr, nperseg=n_fft)
    return frequencies, pxx


def detect_frequency_persistence(audio, sr, center_freq, margin=300, 
                                  time_window=1.0, persistence_threshold=0.5):
    """
    Check if a characteristic frequency has a strong peak relative to nearby frequencies.
    
    This uses a relative peak detection approach: if the target frequency band 
    has a clear peak compared to adjacent frequency bands, it's likely a 
    characteristic machine frequency.
    
    Parameters
    ----------
    audio : numpy array
        Audio signal
    sr : int
        Sample rate
    center_freq : float
        Center frequency to check (Hz)
    margin : float
        Frequency tolerance band (Hz). Detection band is [center-margin, center+margin]
    time_window : float
        Time window to check (seconds, default 1.0 sec)
    persistence_threshold : float
        Minimum ratio of band power to adjacent power (0-2+)
        Default 0.5 means target band should have 50% more power than adjacent bands
    
    Returns
    -------
    dict with keys:
        'is_present': bool - whether frequency peak is detected
        'persistence_score': float - relative strength of target band (0-2+)
        'peak_strength': float - max power in target band
        'confidence': float - overall detection confidence (0-1)
    """
    
    if len(audio) == 0:
        return {
            'is_present': False,
            'persistence_score': 0.0,
            'peak_strength': 0.0,
            'confidence': 0.0
        }
    
    # Limit to requested time window
    n_samples = int(time_window * sr)
    audio_segment = audio[:n_samples]
    
    # Compute power spectrum
    frequencies, pxx = get_frequency_spectrum(audio_segment, sr, n_fft=2048)
    
    # Get overall signal power
    overall_power = np.mean(pxx)
    if overall_power < 1e-10:  # Near silence
        return {
            'is_present': False,
            'persistence_score': 0.0,
            'peak_strength': 0.0,
            'confidence': 0.0
        }
    
    # Find power in target frequency band
    target_mask = (frequencies >= center_freq - margin) & (frequencies <= center_freq + margin)
    target_power = pxx[target_mask]
    peak_strength = np.max(target_power) if len(target_power) > 0 else 0.0
    mean_target_power = np.mean(target_power) if len(target_power) > 0 else 0.0
    
    # Find power in adjacent bands (lower and upper)
    lower_mask = (frequencies >= center_freq - 3*margin) & (frequencies < center_freq - margin)
    upper_mask = (frequencies > center_freq + margin) & (frequencies <= center_freq + 3*margin)
    
    lower_power = np.mean(pxx[lower_mask]) if np.any(lower_mask) else 0
    upper_power = np.mean(pxx[upper_mask]) if np.any(upper_mask) else 0
    adjacent_power = (lower_power + upper_power) / 2
    
    # Relative strength: how much stronger is target band compared to adjacent bands?
    if adjacent_power > 0:
        relative_strength = mean_target_power / adjacent_power
    else:
        relative_strength = mean_target_power / (overall_power + 1e-10)
    
    # Check if it's a clear peak (should be at least 1.5x stronger than adjacent)
    is_clear_peak = relative_strength > 1.0
    
    # Calculate persistence using STFT over time
    stft_matrix = librosa.stft(audio_segment, n_fft=2048, hop_length=512)
    stft_freq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Find time frames where target frequency is strong
    freq_idx_min = np.argmin(np.abs(stft_freq - (center_freq - margin)))
    freq_idx_max = np.argmin(np.abs(stft_freq - (center_freq + margin)))
    freq_range = np.arange(freq_idx_min, min(freq_idx_max + 1, stft_matrix.shape[0]))
    
    # Calculate power in target band at each time frame
    if len(freq_range) > 0:
        band_power_over_time = np.mean(np.abs(stft_matrix[freq_range, :]), axis=0)
        # Threshold: time frames where band power exceeds 40% of max in this frame
        max_power = np.max(band_power_over_time) if len(band_power_over_time) > 0 else 1e-10
        strong_frames = band_power_over_time > (0.4 * max_power)
        persistence_score = np.sum(strong_frames) / len(strong_frames) if len(strong_frames) > 0 else 0.0
    else:
        persistence_score = 0.0
    
    # Confidence: combination of peak clarity and persistence
    confidence = min(1.0, (relative_strength / 2.0) * persistence_score)
    
    return {
        'is_present': is_clear_peak,
        'persistence_score': persistence_score,
        'peak_strength': peak_strength,
        'confidence': confidence
    }


def check_machine_sound(audio, sr, confidence_threshold=0.3):
    """
    Determine if audio contains a known machine sound or is just noise/silence.
    
    Parameters
    ----------
    audio : numpy array
        Audio signal
    sr : int
        Sample rate
    confidence_threshold : float
        Minimum confidence to declare a frequency as "present" (0-1)
        Default 0.3 = 30%
    
    Returns
    -------
    dict with keys:
        'is_machine_sound': bool - True if any known machine frequency detected
        'has_silence': bool - True if audio is too quiet (RMS < 0.01)
        'best_match': str or None - which category frequency was detected ('normal', 'electrical', 'mechanical')
        'best_confidence': float - confidence of best match (0-1)
        'details': dict - detailed results for each frequency band
    """
    
    # Check for silence first (energy-based)
    rms_energy = np.sqrt(np.mean(audio ** 2))
    has_silence = rms_energy < 0.01
    
    if has_silence:
        return {
            'is_machine_sound': False,
            'has_silence': True,
            'best_match': None,
            'best_confidence': 0.0,
            'details': {}
        }
    
    # Check each known frequency band
    results = {}
    best_match = None
    best_confidence = 0.0
    
    for category, band_info in FREQUENCY_BANDS.items():
        freq_result = detect_frequency_persistence(
            audio, sr,
            center_freq=band_info['center'],
            margin=band_info['margin'],
            time_window=1.0
        )
        results[category] = freq_result
        
        if freq_result['confidence'] > best_confidence:
            best_confidence = freq_result['confidence']
            best_match = category
    
    # Machine sound is present if any frequency band exceeds threshold
    is_machine_sound = best_confidence >= confidence_threshold
    
    return {
        'is_machine_sound': is_machine_sound,
        'has_silence': False,
        'best_match': best_match if is_machine_sound else None,
        'best_confidence': best_confidence,
        'details': results
    }
